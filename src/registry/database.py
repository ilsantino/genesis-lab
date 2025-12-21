"""
SQLite-based registry for synthetic datasets.

Tracks generated datasets, quality metrics, and training runs.

Usage:
    uv run python -m src.registry.database
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generation.schemas import DatasetMetadata, QualityMetrics, BiasMetrics


__all__ = ["DatasetRegistry"]


class DatasetRegistry:
    """
    SQLite-based registry for tracking synthetic datasets.
    
    Provides CRUD operations for datasets, quality metrics, and training runs.
    Enables tracking of dataset lineage and model performance over time.
    
    Example:
        >>> registry = DatasetRegistry()
        >>> dataset_id = registry.register_dataset(metadata)
        >>> registry.update_quality_metrics(dataset_id, quality_metrics)
        >>> datasets = registry.list_datasets(domain="customer_service")
    """
    
    def __init__(self, db_path: str = "data/registry.db"):
        """
        Initialize database and create tables if not exist.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                size INTEGER NOT NULL,
                generation_date TIMESTAMP,
                model_used TEXT,
                file_path TEXT,
                file_format TEXT,
                quality_score REAL,
                bias_detected BOOLEAN DEFAULT 0,
                bias_severity TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create quality_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                completeness REAL,
                consistency REAL,
                realism REAL,
                diversity REAL,
                overall REAL,
                issues_count INTEGER DEFAULT 0,
                warnings_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # Create training_runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                model_type TEXT,
                accuracy REAL,
                f1_score REAL,
                precision_score REAL,
                recall_score REAL,
                training_time_seconds REAL,
                notes TEXT,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_dataset(
        self,
        domain: str,
        size: int,
        file_path: str,
        file_format: str = "json",
        model_used: str = "claude_35_sonnet",
        notes: Optional[str] = None
    ) -> str:
        """
        Register a new dataset in the registry.
        
        Args:
            domain: Dataset domain (e.g., "customer_service", "time_series")
            size: Number of items in the dataset
            file_path: Path to the dataset file
            file_format: File format (json, jsonl, parquet, csv)
            model_used: Model used for generation
            notes: Optional notes about the dataset
        
        Returns:
            Dataset ID (UUID)
        """
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        generation_date = datetime.now(timezone.utc).isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO datasets (
                id, domain, size, generation_date, model_used,
                file_path, file_format, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id, domain, size, generation_date, model_used,
            file_path, file_format, notes
        ))
        
        conn.commit()
        conn.close()
        
        return dataset_id
    
    def register_from_metadata(self, metadata: DatasetMetadata) -> str:
        """
        Register dataset from DatasetMetadata object.
        
        Args:
            metadata: DatasetMetadata schema object
        
        Returns:
            Dataset ID
        """
        return self.register_dataset(
            domain=metadata.domain,
            size=metadata.size,
            file_path=metadata.file_path,
            file_format=metadata.file_format,
            model_used=metadata.model_used,
            notes=metadata.notes
        )
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve dataset by ID.
        
        Args:
            dataset_id: Dataset ID to retrieve
        
        Returns:
            Dataset dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT d.*, 
                   qm.completeness, qm.consistency, qm.realism, 
                   qm.diversity, qm.overall as quality_overall
            FROM datasets d
            LEFT JOIN quality_metrics qm ON d.id = qm.dataset_id
            WHERE d.id = ?
        """, (dataset_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def list_datasets(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all datasets, optionally filtered by domain.
        
        Args:
            domain: Optional domain filter
        
        Returns:
            List of dataset dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if domain:
            cursor.execute("""
                SELECT d.*, 
                       qm.completeness, qm.consistency, qm.realism,
                       qm.diversity, qm.overall as quality_overall
                FROM datasets d
                LEFT JOIN quality_metrics qm ON d.id = qm.dataset_id
                WHERE d.domain = ?
                ORDER BY d.generation_date DESC
            """, (domain,))
        else:
            cursor.execute("""
                SELECT d.*, 
                       qm.completeness, qm.consistency, qm.realism,
                       qm.diversity, qm.overall as quality_overall
                FROM datasets d
                LEFT JOIN quality_metrics qm ON d.id = qm.dataset_id
                ORDER BY d.generation_date DESC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def update_quality_metrics(self, dataset_id: str, metrics: QualityMetrics) -> None:
        """
        Update quality metrics for a dataset.
        
        Args:
            dataset_id: Dataset ID to update
            metrics: QualityMetrics object
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if metrics already exist
        cursor.execute(
            "SELECT id FROM quality_metrics WHERE dataset_id = ?",
            (dataset_id,)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing metrics
            cursor.execute("""
                UPDATE quality_metrics SET
                    completeness = ?,
                    consistency = ?,
                    realism = ?,
                    diversity = ?,
                    overall = ?,
                    issues_count = ?,
                    warnings_count = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE dataset_id = ?
            """, (
                metrics.completeness_score,
                metrics.consistency_score,
                metrics.realism_score,
                metrics.diversity_score,
                metrics.overall_quality_score,
                len(metrics.issues_found),
                len(metrics.warnings),
                dataset_id
            ))
        else:
            # Insert new metrics
            cursor.execute("""
                INSERT INTO quality_metrics (
                    dataset_id, completeness, consistency, realism,
                    diversity, overall, issues_count, warnings_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                metrics.completeness_score,
                metrics.consistency_score,
                metrics.realism_score,
                metrics.diversity_score,
                metrics.overall_quality_score,
                len(metrics.issues_found),
                len(metrics.warnings)
            ))
        
        # Update dataset quality_score
        cursor.execute("""
            UPDATE datasets SET quality_score = ? WHERE id = ?
        """, (metrics.overall_quality_score, dataset_id))
        
        conn.commit()
        conn.close()
    
    def update_bias_metrics(self, dataset_id: str, metrics: BiasMetrics) -> None:
        """
        Update bias information for a dataset.
        
        Args:
            dataset_id: Dataset ID to update
            metrics: BiasMetrics object
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE datasets SET
                bias_detected = ?,
                bias_severity = ?
            WHERE id = ?
        """, (
            1 if metrics.bias_detected else 0,
            metrics.bias_severity,
            dataset_id
        ))
        
        conn.commit()
        conn.close()
    
    def register_training_run(
        self,
        dataset_id: str,
        model_type: str,
        results: Dict[str, Any]
    ) -> str:
        """
        Register a training run with metrics.
        
        Args:
            dataset_id: Dataset used for training
            model_type: Type of model trained (e.g., "logistic_regression", "xgboost")
            results: Dictionary with training results (accuracy, f1_score, etc.)
        
        Returns:
            Training run ID
        """
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO training_runs (
                id, dataset_id, model_type, accuracy, f1_score,
                precision_score, recall_score, training_time_seconds, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            dataset_id,
            model_type,
            results.get("accuracy"),
            results.get("f1_score"),
            results.get("precision"),
            results.get("recall"),
            results.get("training_time"),
            results.get("notes")
        ))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def get_training_history(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Get all training runs for a dataset.
        
        Args:
            dataset_id: Dataset ID
        
        Returns:
            List of training run dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM training_runs
            WHERE dataset_id = ?
            ORDER BY trained_at DESC
        """, (dataset_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset and its associated metrics.
        
        Args:
            dataset_id: Dataset ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT id FROM datasets WHERE id = ?", (dataset_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Delete related records
        cursor.execute("DELETE FROM quality_metrics WHERE dataset_id = ?", (dataset_id,))
        cursor.execute("DELETE FROM training_runs WHERE dataset_id = ?", (dataset_id,))
        cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Count datasets by domain
        cursor.execute("""
            SELECT domain, COUNT(*) as count 
            FROM datasets 
            GROUP BY domain
        """)
        domain_counts = {row["domain"]: row["count"] for row in cursor.fetchall()}
        
        # Total datasets
        cursor.execute("SELECT COUNT(*) as total FROM datasets")
        total_datasets = cursor.fetchone()["total"]
        
        # Total training runs
        cursor.execute("SELECT COUNT(*) as total FROM training_runs")
        total_runs = cursor.fetchone()["total"]
        
        # Average quality score
        cursor.execute("SELECT AVG(quality_score) as avg FROM datasets WHERE quality_score IS NOT NULL")
        avg_quality = cursor.fetchone()["avg"]
        
        conn.close()
        
        return {
            "total_datasets": total_datasets,
            "datasets_by_domain": domain_counts,
            "total_training_runs": total_runs,
            "average_quality_score": avg_quality
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of the registry."""
        stats = self.get_stats()
        datasets = self.list_datasets()
        
        print("\n" + "=" * 60)
        print("DATASET REGISTRY SUMMARY")
        print("=" * 60)
        print(f"\nDatabase: {self.db_path}")
        print(f"Total datasets: {stats['total_datasets']}")
        print(f"Total training runs: {stats['total_training_runs']}")
        
        if stats['average_quality_score']:
            print(f"Average quality score: {stats['average_quality_score']:.1f}/100")
        
        if stats['datasets_by_domain']:
            print("\nDatasets by domain:")
            for domain, count in stats['datasets_by_domain'].items():
                print(f"  - {domain}: {count}")
        
        if datasets:
            print("\n" + "-" * 40)
            print("REGISTERED DATASETS")
            print("-" * 40)
            for ds in datasets[:10]:  # Show first 10
                quality = ds.get('quality_overall') or ds.get('quality_score')
                quality_str = f"{quality:.1f}" if quality else "N/A"
                bias_str = "Yes" if ds.get('bias_detected') else "No"
                print(f"\n  [{ds['id']}]")
                print(f"    Domain: {ds['domain']}")
                print(f"    Size: {ds['size']} items")
                print(f"    Quality: {quality_str}/100")
                print(f"    Bias detected: {bias_str}")
                print(f"    File: {ds['file_path']}")
        
        print("\n" + "=" * 60)


def main():
    """Demonstrate registry CRUD operations."""
    import tempfile
    import os
    
    print("=" * 60)
    print("Dataset Registry Demo")
    print("=" * 60)
    
    # Use temp database for demo
    db_path = "data/registry_demo.db"
    
    # Clean up any existing demo db
    if Path(db_path).exists():
        os.remove(db_path)
    
    print(f"\n1. Creating registry at: {db_path}")
    registry = DatasetRegistry(db_path=db_path)
    print("   [OK] Database initialized")
    
    # Register sample datasets
    print("\n2. Registering sample datasets...")
    
    ds1_id = registry.register_dataset(
        domain="customer_service",
        size=10,
        file_path="data/synthetic/cs_smoke_v2.json",
        file_format="json",
        model_used="claude_35_sonnet",
        notes="Bilingual smoke test (5 EN + 5 ES)"
    )
    print(f"   [OK] Registered: {ds1_id}")
    
    ds2_id = registry.register_dataset(
        domain="time_series",
        size=9,
        file_path="data/synthetic/ts_smoke_v2.json",
        file_format="json",
        model_used="claude_35_sonnet",
        notes="Time series smoke test"
    )
    print(f"   [OK] Registered: {ds2_id}")
    
    # Update with quality metrics
    print("\n3. Adding quality metrics...")
    
    quality_metrics = QualityMetrics(
        completeness_score=1.0,
        consistency_score=1.0,
        realism_score=0.17,
        diversity_score=0.62,
        overall_quality_score=69.7,
        issues_found=[],
        warnings=["Low intent coverage"]
    )
    registry.update_quality_metrics(ds1_id, quality_metrics)
    print(f"   [OK] Quality metrics added to {ds1_id}")
    
    # Update with bias metrics
    print("\n4. Adding bias metrics...")
    
    bias_metrics = BiasMetrics(
        demographic_balance={"language": {"en": 0.5, "es": 0.5}},
        sentiment_distribution={"positive": 0.2, "neutral": 0.5, "negative": 0.3},
        topic_coverage={},
        bias_detected=False,
        bias_severity="none",
        recommendations=[]
    )
    registry.update_bias_metrics(ds1_id, bias_metrics)
    print(f"   [OK] Bias metrics added to {ds1_id}")
    
    # Register training run
    print("\n5. Registering training run...")
    
    run_id = registry.register_training_run(
        dataset_id=ds1_id,
        model_type="logistic_regression",
        results={
            "accuracy": 0.85,
            "f1_score": 0.82,
            "precision": 0.84,
            "recall": 0.80,
            "training_time": 2.5,
            "notes": "Baseline intent classifier"
        }
    )
    print(f"   [OK] Training run registered: {run_id}")
    
    # Retrieve dataset
    print("\n6. Retrieving dataset...")
    
    dataset = registry.get_dataset(ds1_id)
    if dataset:
        print(f"   [OK] Retrieved: {dataset['id']}")
        print(f"        Domain: {dataset['domain']}")
        print(f"        Size: {dataset['size']}")
        print(f"        Quality: {dataset.get('quality_score', 'N/A')}")
    
    # Get training history
    print("\n7. Getting training history...")
    
    history = registry.get_training_history(ds1_id)
    print(f"   [OK] Found {len(history)} training runs")
    for run in history:
        print(f"        - {run['model_type']}: accuracy={run['accuracy']:.2f}")
    
    # List all datasets
    print("\n8. Listing all datasets...")
    
    all_datasets = registry.list_datasets()
    print(f"   [OK] Found {len(all_datasets)} datasets")
    
    # Filter by domain
    cs_datasets = registry.list_datasets(domain="customer_service")
    print(f"   [OK] Customer service datasets: {len(cs_datasets)}")
    
    # Print summary
    registry.print_summary()
    
    print("\n[OK] Demo complete!")
    print(f"     Database saved at: {db_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

