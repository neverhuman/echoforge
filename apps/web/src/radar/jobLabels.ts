import type { JobOption, JobSummary } from './radarContract';

export type JobSelection = 'pipeline' | 'suite';

export function coerceSelection(selection: string): JobSelection {
  return selection === 'suite' ? 'suite' : 'pipeline';
}

export function pipelineLabel(pipelineId: string, pipelines: JobOption[]) {
  return pipelines.find((pipeline) => pipeline.id === pipelineId)?.label ?? pipelineId;
}

export function requestLabel(request: JobSummary['request'], pipelines: JobOption[]) {
  return request.selection === 'pipeline'
    ? pipelineLabel(request.pipeline_id, pipelines)
    : request.suite_id;
}
