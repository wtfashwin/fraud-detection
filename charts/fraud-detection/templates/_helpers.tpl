{{/*
Expand the name of the chart.
*/}}
{{- define "fraud-detection.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "fraud-detection.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "fraud-detection.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "fraud-detection.labels" -}}
helm.sh/chart: {{ include "fraud-detection.chart" . }}
{{ include "fraud-detection.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "fraud-detection.selectorLabels" -}}
app.kubernetes.io/name: {{ include "fraud-detection.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "fraud-detection.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "fraud-detection.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/* One explicitly selected artifact for API, init containers and worker. */}}
{{- define "fraud-detection.image" -}}
{{- printf "%s:%s" .Values.image.repository (required "Set image.tag to the explicitly built API/worker artifact tag" .Values.image.tag) -}}
{{- end -}}

{{- define "fraud-detection.redisHost" -}}
{{- if or (ne .Values.redis.architecture "standalone") .Values.redis.sentinel.enabled .Values.redis.auth.enabled -}}
{{- fail "This baseline supports standalone Redis without auth only; authenticated or Sentinel deployments require explicit broker/scaler configuration" -}}
{{- end -}}
{{- printf "%s-master" (include "common.names.fullname" .Subcharts.redis) -}}
{{- end -}}

{{- define "fraud-detection.environment" -}}
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ include "fraud-detection.fullname" . }}-db
      key: DATABASE_URL
- name: CELERY_BROKER_URL
  value: {{ printf "redis://%s:6379/0" (include "fraud-detection.redisHost" .) | quote }}
- name: MODEL_DIR
  value: {{ .Values.modelDir | quote }}
{{- if .Values.mlflowEnabled }}
- name: MLFLOW_TRACKING_URI
  value: {{ printf "http://%s:%v" (include "mlflow.fullname" .Subcharts.mlflow) .Values.mlflow.service.port | quote }}
{{- end }}
{{- if .Values.otel.enabled }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ required "Set otel.endpoint when otel.enabled=true" .Values.otel.endpoint | quote }}
{{- end }}
{{- end -}}
