"""initial
Revision ID: 0001_initial_transaction_results
Revises: 
Create Date: 2025-11-05 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg

# revision identifiers, used by Alembic.
revision = '0001_initial_transaction_results'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'transaction_results',
        sa.Column('id', pg.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('input_data', pg.JSONB, nullable=False),
        sa.Column('shap_values', pg.JSONB, nullable=True),
        sa.Column('prediction_score', sa.Float, nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
    )


def downgrade():
    op.drop_table('transaction_results')
