import time
import uuid
from kubernetes import client, config
from kubernetes.client import V1EnvVar
from kubernetes.stream import stream

def create_interactive_job(
    namespace: str,
    image: str,
    command: list[str] = None,
    env: dict[str, str] = None,
) -> str:
    """
    Create a new Job with a random name for interactive usage.
    Returns the generated job_name.
    """
    config.load_incluster_config()
    batch_v1 = client.BatchV1Api()

    job_name = f"tool-job-{str(uuid.uuid4())[:8]}"

    env_list = []
    if env:
        env_list = [V1EnvVar(name=k, value=v) for k, v in env.items()]

    job_body = client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name),
        spec=client.V1JobSpec(
            parallelism=1,
            completions=1,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"job-name": job_name}),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="tool-container",
                            image=image,
                            tty=True,
                            stdin=True,
                            command=command,  # e.g. ["sh", "-c", "while true; do ..."]
                            env=env_list,
                        )
                    ],
                ),
            ),
        ),
    )

    batch_v1.create_namespaced_job(
        namespace=namespace,
        body=job_body,
    )
    return job_name

def get_pod_for_job(job_name: str, namespace: str, timeout=30) -> str:
    """
    Find the Pod that was created by the Job (assuming parallelism=1).
    Returns the pod name once it's found, or raises if not found in time.
    """
    config.load_incluster_config()
    core_v1 = client.CoreV1Api()

    label_selector = f"job-name={job_name}"
    start_time = time.time()

    while True:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
        )
        if pods.items:
            # assuming only one Pod for this Job
            return pods.items[0].metadata.name

        if (time.time() - start_time) > timeout:
            raise RuntimeError(f"Timed out waiting for Pod of Job {job_name}")
        time.sleep(1)

def attach_to_pod(pod_name: str, namespace: str):
    """
    Attach to a single Pod's container (named 'tool-container') for interactive I/O.
    Very similar to 'kubectl attach -it <pod>'.
    """
    config.load_incluster_config()
    core_v1 = client.CoreV1Api()

    attach_conn = stream(
        core_v1.connect_get_namespaced_pod_attach,
        name=pod_name,
        namespace=namespace,
        container="tool-container",
        stdin=True,
        stdout=True,
        stderr=True,
        tty=True,
        _preload_content=False
    )

    # Here you would implement the same pattern as in the previous example:
    # - spawn threads to read from attach_conn (stdout/stderr) and write to local sys.stdout
    # - read from sys.stdin and write to attach_conn (stdin).
    # For brevity, we’ll just show a simplified version:

    import sys
    import threading

    def read_stdout():
        while attach_conn.is_open():
            if attach_conn.peek_stdout():
                chunk = attach_conn.read_stdout()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            if attach_conn.peek_stderr():
                chunk = attach_conn.read_stderr()
                if chunk:
                    sys.stderr.write(chunk)
                    sys.stderr.flush()

    def write_stdin():
        while attach_conn.is_open():
            user_input = sys.stdin.read(1)
            if not user_input:  # EOF
                break
            attach_conn.write_stdin(user_input)

    t_out = threading.Thread(target=read_stdout, daemon=True)
    t_in = threading.Thread(target=write_stdin, daemon=True)
    t_out.start()
    t_in.start()

    t_out.join()
    t_in.join()

    attach_conn.close()
    print("Attach session finished.")


def delete_job(job_name: str, namespace: str):
    config.load_incluster_config()
    batch_v1 = client.BatchV1Api()

    # Foreground or Background depends on whether you want to wait for Pod cleanup
    propagation_policy = "Foreground"
    batch_v1.delete_namespaced_job(
        name=job_name,
        namespace=namespace,
        body=client.V1DeleteOptions(propagation_policy=propagation_policy),
    )
    print(f"Job {job_name} is being deleted with {propagation_policy} propagation.")

