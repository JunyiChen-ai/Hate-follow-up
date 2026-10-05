"""Continuous pixel occurrences with bounded mismatch observations and lookup."""
from pathlib import Path
import time
import numpy as np
from tracking import SPEC,advance,cut,crop,iou
from inputs import ROOT,pixel_box,save_rgb


class Controller:
    def __init__(self,folder,windows,observe,save=save_rgb):
        self.folder=Path(folder);self.windows=windows;self.observe=observe
        self.save=save
        self.events=[];self.active={};self.repairs=[];self.used={}
        self.previous=None;self.previous_entry=None;self.processing_seconds=0.

    def end(self,identity,reason,time_now):
        record=self.active.pop(identity)
        self.flush(record)
        record.pop('_reference',None)
        record['ending']=dict(reason=reason,observed_at=time_now)

    def flush(self,record):
        pending=record.pop('_pending_last',None)
        if pending is not None:self.save(pending['rgb'],ROOT/pending['path'])

    def support(self,identity,entry,rgb,box,reason,source_ref=None):
        record=self.events[identity];w=min(int(entry['time']//SPEC['window_seconds']),len(self.windows)-1)
        point=dict(index=entry['index'],time=entry['time'],box=box,reason=reason)
        if source_ref is not None:point['source_ref']=source_ref
        if record['supports'] and record['supports'][-1]['index']==entry['index']:
            assert record['supports'][-1]['box']==box
            if source_ref is not None:record.setdefault('additional_observations',[]).append(dict(index=entry['index'],source_ref=source_ref))
            return
        record['supports'].append(point);record['last_box']=box
        relative=f'occurrence_{identity:06d}/window_{w:05d}'
        endpoints=record['windows'].setdefault(str(w),{})
        if record.get('_pending_last',{}).get('window')!=w:self.flush(record)
        pixels=crop(rgb,box)
        if 'first' not in endpoints:
            path=self.folder/'crops'/(relative+'_first.png');self.save(pixels,path)
            endpoints['first']={**point,'path':str(path.relative_to(ROOT))}
        path=self.folder/'crops'/(relative+'_last.png')
        endpoints['last']={**point,'path':str(path.relative_to(ROOT))}
        record['_pending_last']=dict(window=w,rgb=pixels,path=str(path.relative_to(ROOT)))

    def install(self,observation,entry,rgb,source_ref):
        if observation['status']!='TEXT':return None
        box=pixel_box(observation['box'],[rgb.shape[1],rgb.shape[0]])
        for identity,record in sorted(self.active.items()):
            if record['supports'][-1]['index']!=entry['index']:continue
            if record['text']==observation['text'] and iou(record['last_box'],box)>=SPEC['same_region_iou']:
                self.support(identity,entry,rgb,record['last_box'],'fresh_observation',source_ref);return identity
        identity=len(self.events)
        record=dict(id=identity,text=observation['text'],initial_source=source_ref,
            initial_index=entry['index'],initial_time=entry['time'],initial_box=box,
            supports=[],windows={},tracking=[],ending=None)
        self.events.append(record);self.active[identity]=record
        record['_reference']=crop(rgb,box)
        self.support(identity,entry,rgb,box,'fresh_observation',source_ref)
        while len(self.active)>SPEC['max_active_tracks']:
            old=min(self.active,key=lambda k:(self.active[k]['initial_time'],k))
            self.end(old,'capacity_unknown',entry['time'])
        return identity

    def step(self,entry,rgb,anchors):
        start=time.perf_counter();now=entry['time'];failures=[];candidates={}
        global_reason=None
        if self.previous is not None:
            if now-self.previous_entry['time']>SPEC['max_gap_seconds']:global_reason='pts_gap'
            else:
                is_cut,cut_evidence=cut(self.previous,rgb)
                if is_cut:global_reason='cut'
            for identity,record in list(self.active.items()):
                if global_reason:
                    box=None;evidence=dict(reason=global_reason)
                    if global_reason=='cut':evidence['cut']=cut_evidence
                else:box,evidence=advance(self.previous,rgb,record['last_box'],record['_reference'])
                record['tracking'].append(dict(from_index=self.previous_entry['index'],to_index=entry['index'],time=now,evidence=evidence))
                if box is None:
                    failures.append(dict(id=identity,old_box=record['last_box'],evidence=evidence))
                    self.end(identity,evidence['reason'],now)
                else:candidates[identity]=box
        # A fresh literal change invalidates current-frame support of old words.
        # Previous accepted points remain historical observations unchanged.
        for observation,source_ref in anchors:
            if observation['status']!='TEXT':continue
            actual_box=pixel_box(observation['box'],[rgb.shape[1],rgb.shape[0]])
            for identity,box in list(candidates.items()):
                if iou(box,actual_box)>=SPEC['same_region_iou'] and self.active[identity]['text']!=observation['text']:
                    self.end(identity,'fresh_literal_change',now);candidates.pop(identity)
        for identity,box in candidates.items():self.support(identity,entry,rgb,box,'pixel_correspondence')
        for observation,source_ref in anchors:self.install(observation,entry,rgb,source_ref)
        w=min(int(now//SPEC['window_seconds']),len(self.windows)-1)
        for failure in sorted(failures,key=lambda f:f['id']):
            if self.used.get(w,0)>=SPEC['max_repairs_per_window']:
                self.repairs.append(dict(window=w,time=now,index=entry['index'],failed_id=failure['id'],reason='budget_unknown'))
                continue
            self.used[w]=self.used.get(w,0)+1
            observation,source_ref=self.observe(w,entry,rgb,failure,len(self.repairs))
            self.repairs.append(dict(window=w,time=now,index=entry['index'],failed_id=failure['id'],reason='fresh_observation',source_ref=source_ref))
            self.install(observation,entry,rgb,source_ref)
        self.previous=rgb;self.previous_entry=entry
        self.processing_seconds+=time.perf_counter()-start

    def finish(self):
        end=self.previous_entry['time'] if self.previous_entry is not None else 0.
        for identity in list(self.active):self.end(identity,'end_of_input',end)
        for record in self.events:record.pop('_reference',None)
        return dict(events=self.events,repairs=self.repairs,processing_seconds=self.processing_seconds)


def interval_lookup(events,window):
    i=window['i'];a,b=window['start'],window['end'];result=[]
    for event in events:
        local=[point for point in event['supports'] if a<=point['time']<b]
        if not local:continue
        endpoints=event['windows'][str(i)]
        assert endpoints['first']['index']==local[0]['index'] and endpoints['last']['index']==local[-1]['index']
        result.append(dict(id=event['id'],text=event['text'],first=endpoints['first'],last=endpoints['last'],
            actual_supported_times=[p['time'] for p in local],actual_supported_indices=[p['index'] for p in local]))
    result.sort(key=lambda r:(r['first']['time'],r['id']))
    return result[:SPEC['max_lookup_events']]
