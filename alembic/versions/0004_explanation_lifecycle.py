"""Create the explanation table used by both the API and Celery worker.

Revision ID: 0004_explanation_lifecycle
Revises: fbae492048d4
"""
from alembic import op, context
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_explanation_lifecycle"
down_revision = "fbae492048d4"
branch_labels = None
depends_on = None


def upgrade():
    # Earlier API versions created this table at startup outside migrations.
    bind = op.get_bind()
    if context.is_offline_mode() or not sa.inspect(bind).has_table("shap_explanations"):
        op.create_table(
            "shap_explanations",
            sa.Column("transaction_id", sa.String(128), primary_key=True),
            sa.Column("correlation_id", sa.String(64), nullable=False),
            sa.Column("status", sa.String(16), nullable=False, server_default="PENDING"),
            sa.Column("shap_values", postgresql.JSONB(), nullable=True),
            sa.Column("feature_names", postgresql.JSONB(), nullable=True),
            sa.Column("base_value", sa.Float(), nullable=True),
            sa.Column("explanation_method", sa.String(64), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
    else:
        # Adopt pre-migration explanations without overwriting their completed data.
        op.add_column("shap_explanations", sa.Column("status", sa.String(16), nullable=False, server_default="COMPLETED"))
        op.add_column("shap_explanations", sa.Column("base_value", sa.Float(), nullable=True))
        op.add_column("shap_explanations", sa.Column("explanation_method", sa.String(64), nullable=True))
        op.alter_column("shap_explanations", "shap_values", nullable=True)
        op.alter_column("shap_explanations", "status", server_default="PENDING")


def downgrade():
    op.drop_table("shap_explanations")
