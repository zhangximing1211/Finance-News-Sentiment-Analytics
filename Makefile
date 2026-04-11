PYTHON ?= python3

.PHONY: audit-data train-baseline evaluate-baseline train train-bert evaluate-bert test api web tree process-review-queue review-queue-digest daily-report weekly-report feedback-loop-maintenance

audit-data:
	$(PYTHON) services/trainer/scripts/prepare_data.py

train:
	$(PYTHON) services/trainer/scripts/train.py

train-baseline:
	$(PYTHON) services/trainer/scripts/train_baseline.py

evaluate-baseline:
	$(PYTHON) services/trainer/scripts/evaluate.py --split test

train-bert:
	$(PYTHON) services/trainer/scripts/train_bert.py

evaluate-bert:
	$(PYTHON) services/trainer/scripts/train_bert.py --evaluate-only --split test

test:
	$(PYTHON) -m unittest discover -s tests

api:
	uvicorn apps.api.app.main:app --reload --port 8000

process-review-queue:
	$(PYTHON) services/worker/jobs/process_review_queue.py

review-queue-digest:
	$(PYTHON) services/worker/jobs/daily_digest.py

daily-report:
	$(PYTHON) services/worker/jobs/generate_agent_report.py --report-type daily

weekly-report:
	$(PYTHON) services/worker/jobs/generate_agent_report.py --report-type weekly

feedback-loop-maintenance:
	$(PYTHON) services/worker/jobs/run_feedback_loop_maintenance.py --report-type weekly

web:
	npm --prefix apps/web run dev

tree:
	find . -maxdepth 3 -type d | sort
