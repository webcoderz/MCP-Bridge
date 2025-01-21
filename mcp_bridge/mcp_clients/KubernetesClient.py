import asyncio

from mcp_bridge.mcp_clients.session import McpClientSession
from mcp_bridge.config import config
from mcpx.client.transports.kubernetes import KubernetesMCPServer, create_interactive_job, get_pod_for_job
from .AbstractClient import GenericMcpClient
from loguru import logger


class KubernetesClient(GenericMcpClient):
    config: KubernetesMCPServer

    def __init__(self, name: str, config: KubernetesMCPServer) -> None:
        super().__init__(name=name)

        self.config = config

    def _maintain_session(self):
        with create_interactive_job(self.config) as client:
            logger.debug(f"made instance of docker client for {self.name}")
            with McpClientSession(*client) as session:
                session.initialize()
                logger.debug(f"finished initialise session for {self.name}")
                self.session = session

                try:
                    while True:
                        if config.logging.log_server_pings:
                            logger.debug(f"pinging session for {self.name}")

                        session.send_ping()

                except Exception as exc:
                    logger.error(f"ping failed for {self.name}: {exc}")
                    self.session = None

        logger.debug(f"exiting session for {self.name}")
