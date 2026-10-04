"""Independent partition/representative oracle and real PTS fixture checks."""
from pathlib import Path
import sys
import warnings
from types import SimpleNamespace
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
import tree


def test_hierarchy():
    angles=np.linspace(0,2.5,40);features=np.c_[np.cos(angles),np.sin(angles)].astype(np.float32)
    entries=[dict(index=3*i,time=float(i)) for i in range(40)];calls=[]
    def cap(i):calls.append(i);return dict(text=f'observation {i}',tokens=[i])
    result=tree.Builder(features,entries,cap,lambda roots:[3,3,2,1]).build()
    assert len(result['rounds'])==1 and len(result['nodes'])==18
    assert len(calls)==len(set(calls))==len(result['captions'])
    nodes={n['id']:n for n in result['nodes']}
    for n in nodes.values():
        members=n['members'];centroid=sum((features[i] for i in members),np.zeros(2,dtype=np.float32))/len(members)
        distances=[float(np.dot(features[i]-centroid,features[i]-centroid)) for i in members]
        winner=min(range(len(members)),key=lambda k:(distances[k],entries[members[k]]['time'],entries[members[k]]['index']))
        assert n['representative']==members[winner]
        kids=[c for c in nodes.values() if c['parent']==n['id']]
        if kids:assert sorted(i for c in kids for i in c['members'])==sorted(members)
    for a in range(0,40,8):
        packet=tree.window_packet(result,features,entries,a,a+8)
        assert 1<=len(packet['pool_members'])<=2
        assert all(a<=entries[i]['time']<a+8 for i in packet['pool_members'])
        assert len(set(packet['ancestor_ids']))==len(packet['ancestor_ids'])
        for leaf in packet['leaf_ids']:
            p=nodes[leaf]['parent']
            while p:
                assert p in packet['ancestor_ids'];p=nodes[p]['parent']
    assert 'parent=ROOT' in tree.tree_text(result)


def test_identical_singleton_and_breadth():
    x=np.tile([1.,0.],(20,1)).astype(np.float32);entries=[dict(index=i,time=float(i)) for i in range(20)]
    calls=[]
    def cap(i):calls.append(i);return dict(text=str(i),tokens=[])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=tree.Builder(x,entries,cap,lambda r:[3]*len(r)).build()
    assert [r['requested_width'] for r in result['rounds']]==[4,8]
    assert len(result['nodes'])==1 and calls==[0]
    one=tree.Builder(x[:1],entries[:1],cap,lambda r:[3]).build();assert len(one['nodes'])==1
    assert tree.Builder(x[:0],[],cap,lambda r:[]).build()['nodes']==[]
    distinct=np.eye(20,dtype=np.float32)
    result=tree.Builder(distinct,entries,cap,lambda r:[1]*len(r)).build()
    assert [r['requested_width'] for r in result['rounds']]==[4,8,16]
    assert len(result['nodes'])==16 and all(n['parent'] is None for n in result['nodes'])


def write_video(path,count):
    import av
    container=av.open(str(path),'w');stream=container.add_stream('libx264',rate=8)
    stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
    for i in range(count):
        rgb=np.zeros((64,64,3),dtype=np.uint8);rgb[:]=[40+5*i,120,200-3*i]
        frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
        for packet in stream.encode(frame):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
    container.close()


def test_real_pts_and_coverage():
    folder=ROOT/'runs/20261004_m1_tree/cpu_checks/media_fixture';folder.mkdir(parents=True,exist_ok=True)
    original=tree.image_features
    def fixture_features(j,images):
        v=np.asarray([np.asarray(im).mean((0,1))[:2] for im in images],dtype=np.float32)
        return v/np.linalg.norm(v,axis=1,keepdims=True)
    tree.image_features=fixture_features
    fake=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))))
    try:
        path=folder/'four_seconds.mp4';write_video(path,32)
        features,meta=tree.decode_pool(fake,dict(dataset='Fixture',video_id=path.stem,video_path=str(path),duration=4.))
        assert [e['index'] for e in meta['entries']]==[4,12,20,28]
        np.testing.assert_array_equal([e['time'] for e in meta['entries']],[.5,1.5,2.5,3.5])
        assert features.shape==(4,2) and meta['coverage_repairs']==[]
        witness=tree.Witness(meta,folder/'witnesses');image,dest=witness.image(meta['entries'][2])
        assert dest.is_file() and image.size==(64,64);image.close();witness.close()
        path=folder/'tiny.mp4';write_video(path,2)
        f,m=tree.decode_pool(fake,dict(dataset='Fixture',video_id=path.stem,video_path=str(path),duration=.25))
        assert len(m['entries'])==1 and m['entries'][0]['index']==1 and m['entries'][0]['time']==.125
        assert len(m['coverage_repairs'])==1 and m['windows_without_decoded_frame']==[]
    finally:tree.image_features=original


if __name__=='__main__':
    for test in (test_hierarchy,test_identical_singleton_and_breadth,test_real_pts_and_coverage):
        test();print(test.__name__+' PASS',flush=True)
