"""
Kubeflow Pipeline: Spleen Segmentation
Process 4 patients through: load_data -> preprocess -> inference -> visualize
"""

from kfp import dsl
from kfp import compiler
from kubernetes.client import V1Volume, V1PersistentVolumeClaimVolumeSource, V1VolumeMount

# Base image built from Dockerfile
BASE_IMAGE = "spleen-pipeline:v1"

# Patient IDs to process
PATIENTS = ["spleen_12", "spleen_19", "spleen_29", "spleen_9"]


@dsl.container_component
def load_data_component(patient_id: str):
    """Load CT scan from test dataset"""
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "/app/components/load_data.py"],
        args=[patient_id]
    )


@dsl.container_component
def preprocess_component(patient_id: str):
    """Preprocess CT scan"""
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "/app/components/preprocess.py"],
        args=[patient_id]
    )


@dsl.container_component
def inference_component(patient_id: str):
    """Run segmentation inference"""
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "/app/components/inference.py"],
        args=[patient_id]
    )


@dsl.container_component
def visualize_component(patient_id: str):
    """Create 3-view visualizations"""
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "/app/components/visualize.py"],
        args=[patient_id]
    )


@dsl.pipeline(
    name="Spleen Segmentation Pipeline - Parallel",
    description="Spleen CT segmentation running 4 patients in PARALLEL: load_data -> preprocess -> inference -> visualize"
)
def spleen_pipeline():
    """Main pipeline: Process all 4 patients in PARALLEL (faster execution)"""

    # Define PVC volume
    data_volume = V1Volume(
        name='data-volume',
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name='data-pvc'
        )
    )

    data_volume_mount = V1VolumeMount(
        name='data-volume',
        mount_path='/mnt/data'
    )

    # Process all patients in parallel
    for patient_id in PATIENTS:
        print(f"Setting up pipeline for {patient_id}")

        # Step 1: Load Data
        load_task = load_data_component(patient_id=patient_id)
        load_task.set_caching_options(False)
        load_task.add_pvolumes({'/mnt/data': data_volume})

        # Step 2: Preprocess (depends on THIS patient's load_data only)
        preprocess_task = preprocess_component(patient_id=patient_id)
        preprocess_task.after(load_task)
        preprocess_task.set_caching_options(False)
        preprocess_task.add_pvolumes({'/mnt/data': data_volume})

        # Step 3: Inference (depends on THIS patient's preprocess only)
        inference_task = inference_component(patient_id=patient_id)
        inference_task.after(preprocess_task)
        inference_task.set_caching_options(False)
        inference_task.add_pvolumes({'/mnt/data': data_volume})

        # Step 4: Visualize (depends on THIS patient's inference only)
        visualize_task = visualize_component(patient_id=patient_id)
        visualize_task.after(inference_task)
        visualize_task.set_caching_options(False)
        visualize_task.add_pvolumes({'/mnt/data': data_volume})


if __name__ == "__main__":
    # Compile pipeline to YAML
    compiler.Compiler().compile(
        pipeline_func=spleen_pipeline,
        package_path="spleen_pipeline.yaml"
    )

    print("=" * 60)
    print("Pipeline compiled successfully!")
    print("Output: spleen_pipeline.yaml")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Build Docker image:")
    print("   eval $(minikube docker-env)")
    print("   docker build -t spleen-pipeline:v1 .")
    print("\n2. Deploy PersistentVolume:")
    print("   kubectl apply -f pv.yaml")
    print("\n3. Upload spleen_pipeline.yaml to Kubeflow UI")
    print("=" * 60)
