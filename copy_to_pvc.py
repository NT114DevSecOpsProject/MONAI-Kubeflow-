#!/usr/bin/env python3
"""Copy files to Kubernetes PVC via kubectl exec"""
import subprocess
import sys

def copy_file_to_pod(local_path, pod_name, namespace, remote_path):
    """Copy a local file to a pod using kubectl exec and stdin"""
    print(f"Copying {local_path} to {pod_name}:{remote_path}...")

    # Read local file in binary mode
    with open(local_path, 'rb') as f:
        file_data = f.read()

    # Use kubectl exec to write to remote file
    cmd = [
        'kubectl', 'exec', '-i', pod_name,
        '-n', namespace,
        '--', 'sh', '-c', f'cat > {remote_path}'
    ]

    result = subprocess.run(cmd, input=file_data, capture_output=True)

    if result.returncode == 0:
        print(f"[OK] Successfully copied {len(file_data)} bytes")
        return True
    else:
        print(f"[ERROR] Failed: {result.stderr.decode()}")
        return False

if __name__ == '__main__':
    pod_name = 'data-loader'
    namespace = 'kubeflow'

    # List of files to copy
    files_to_copy = ['spleen_12', 'spleen_19', 'spleen_29', 'spleen_9']

    failed = []
    success = []

    for spleen_id in files_to_copy:
        local_file = rf'E:\monai-kubeflow-demo\monai-demo\test_data\Task09_Spleen\imagesTr\{spleen_id}.nii.gz'
        remote_file = f'/mnt/data/test_data/Task09_Spleen/imagesTr/{spleen_id}.nii.gz'

        print(f"\n{'='*60}")
        print(f"Processing: {spleen_id}")
        print(f"{'='*60}")

        if copy_file_to_pod(local_file, pod_name, namespace, remote_file):
            success.append(spleen_id)
        else:
            failed.append(spleen_id)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Success: {len(success)}/{len(files_to_copy)}")
    for s in success:
        print(f"  - {s}")

    if failed:
        print(f"\n[ERROR] Failed: {len(failed)}/{len(files_to_copy)}")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n[OK] All files copied successfully!")
        sys.exit(0)
