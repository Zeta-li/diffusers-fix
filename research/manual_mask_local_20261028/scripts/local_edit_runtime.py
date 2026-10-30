import copy
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch

from budget_r25_profiles import make_budget_plan, validate_budget_plan
from refresh_policy import ACTIONS, load_head
from trainable_budget_executor import TrainableBudgetExecutor


@dataclass(frozen=True)
class LocalEditRequest:
    request_key: str
    prompt: str
    source_wh: tuple
    roi_xyxy: tuple
    output_hw: tuple = (1024, 1024)
    steps: int = 8
    cfg: float = 1.0

    def as_case(self):
        if not self.request_key or not self.prompt or len(self.source_wh) != 2 or len(self.roi_xyxy) != 4:
            raise ValueError('Request key, prompt, source size and four ROI coordinates are required')
        if self.output_hw != (1024, 1024) or self.steps != 8 or self.cfg != 1.0:
            raise ValueError('Frozen runtime supports1024x1024,8steps,CFG1 only')
        w,h = self.source_wh
        x0,y0,x1,y1 = self.roi_xyxy
        if not all(math.isfinite(v) for v in (w,h,x0,y0,x1,y1)) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('ROI must be finite, nonempty and inside the source image')
        return dict(case_id=self.request_key,prompt=self.prompt,source_wh=list(self.source_wh),bbox=list(self.roi_xyxy))


def request_from_case(case):
    return LocalEditRequest(case['case_id'],case['prompt'],tuple(case['source_wh']),tuple(case['bbox']))


def timing_plan(case, action):
    if action not in ACTIONS and action != 'B':
        raise ValueError('Unknown timing action')
    if action in ('B','R25'):
        return make_budget_plan(case,action)
    plan = make_budget_plan(case,'B')
    middle = 3 if action == 'R4' else 5
    plan.update(kind='budget_'+action,profile=action,full_steps_zero_based=[0,1,middle,7])
    plan['selected_indices_per_step'][middle] = list(range(plan['target_count']))
    plan['target_tokens_per_step'] = list(map(len,plan['selected_indices_per_step']))
    plan['total_target_token_updates'] = sum(plan['target_tokens_per_step'])
    validate_budget_plan(plan)
    return plan


class GeometryRefreshSelector:
    def __init__(self, checkpoint):
        self.head,self.data = load_head(checkpoint,'cpu')
        if self.data['kind'] != 'geometry':
            raise ValueError('This entry point only supports the frozen geometry head')
        self.head.eval().requires_grad_(False)

    @torch.no_grad()
    def select(self, case, r25_plan):
        x0,y0,x1,y1 = case['bbox']; w,h = case['source_wh']
        geometry = torch.tensor([x0/w,y0/h,(x1-x0)/w,(y1-y0)/h,(x1-x0)*(y1-y0)/(w*h),
            len(r25_plan['core_indices'])/4096,r25_plan['effective_local_fraction'],min(len(case['prompt']),512)/512])
        norm = ((geometry-self.data['mean'][:8])/self.data['std'][:8]).clamp(-5,5)
        scores = self.head(norm)
        return ACTIONS[int(scores.argmin())],dict(geometry=geometry.tolist(),scores=scores.tolist(),decision_stage='before_DiT')


def resolve_request(request, profile='R25', selector=None):
    case = request.as_case()
    if profile == 'F':
        return None,dict(request_key=request.request_key,profile='F',resolved_action='F')
    metadata = dict(request_key=request.request_key,profile=profile,hard_output_mask=False,pixel_copy=False)
    if profile == 'SUP':
        if selector is None:raise ValueError('SUP requires frozen selector weights')
        action,detail = selector.select(case,timing_plan(case,'R25'))
        metadata.update(detail)
    else:
        action=profile
    plan=timing_plan(case,action)
    metadata.update(resolved_action=action,total_target_token_updates=plan['total_target_token_updates'],
        full_steps_one_based=[i+1 for i in plan['full_steps_zero_based']],target_tokens_per_step=plan['target_tokens_per_step'])
    return plan,metadata


def export_backend_plan(plan, metadata):
    if plan is None:return dict(schema_version=1,metadata=copy.deepcopy(metadata),dense=True)
    validate_budget_plan(plan)
    return dict(schema_version=1,metadata=copy.deepcopy(metadata),dense=False,steps=8,
        target_tokens=plan['target_count'],source_tokens=plan['target_count'],grid_hw=plan['grid_hw'],packed_channels=plan['packed_channels'],
        full_steps_zero_based=plan['full_steps_zero_based'],selected_target_indices=plan['selected_indices_per_step'],
        local_policy=dict(target_queries='selected_only',target_KV='selected_fresh_outside_cached',source_KV='cached',text_KV='cached',
            inactive_velocity='last_computed',target_latent_update='all_target_rows_Euler',student_adapter='enabled'),
        full_policy=dict(target_queries='all',source_text='fresh',student_adapter='disabled',lightning='enabled'),
        cache_scope='single_request',output_hard_mask=False)


@contextmanager
def install_local_edit(pipe, adapters, request, profile='R25', selector=None):
    model=pipe.transformer
    if getattr(model,'_local_edit_controller_active',False):raise RuntimeError('Nested local edit controller')
    plan,metadata=resolve_request(request,profile,selector)
    original=model.forward
    executor=None if plan is None else TrainableBudgetExecutor(model,plan)
    model._local_edit_controller_active=True
    record=dict(metadata=metadata,backend_plan=export_backend_plan(plan,metadata),completed=False)
    def forward(*args,**kwargs):
        if args:raise ValueError('Expected named transformer fields')
        enabled=executor is not None and executor.step not in executor.full_steps
        with adapters.mode(enabled):
            return original(**kwargs) if executor is None else executor.forward(**kwargs)
    model.forward=forward
    try:
        yield record
        if executor is not None and executor.step!=8:raise RuntimeError('Incomplete8step request')
        record['completed']=True
    finally:
        model.forward=original
        model._local_edit_controller_active=False
        if executor is not None:executor.clear()
