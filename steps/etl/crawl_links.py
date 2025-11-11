from urllib.parse import urlparse

import mlflow
from loguru import logger
from tqdm import tqdm

from llm_pipeline_system.application.crawlers.dispatcher import CrawlerDispatcher
from llm_pipeline_system.domain.documents import UserDocument


def crawl_links(user: UserDocument, links: list[str]) -> list[str]:
    dispatcher = CrawlerDispatcher.build().register_linkedin().register_medium().register_github()

    logger.info(f"Starting to crawl {len(links)} link(s).")

    metadata = {}
    successfull_crawls = 0
    for link in tqdm(links):
        successfull_crawl, crawled_domain = _crawl_link(dispatcher, link, user)
        successfull_crawls += successfull_crawl

        metadata = _add_to_metadata(metadata, crawled_domain, successfull_crawl)

    # Log metadata to MLflow
    try:
        mlflow.log_dict(metadata, "crawled_links_metadata.json")
        mlflow.log_metric("successful_crawls", successfull_crawls)
        mlflow.log_metric("total_links", len(links))
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")
        # Continue execution even if MLflow logging fails

    logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} links.")

    return links


def _crawl_link(dispatcher: CrawlerDispatcher, link: str, user: UserDocument) -> tuple[bool, str]:
    crawler = dispatcher.get_crawler(link)
    crawler_domain = urlparse(link).netloc

    try:
        crawler.extract(link=link, user=user)

        return (True, crawler_domain)
    except Exception as e:
        logger.error(f"An error occurred while crowling: {e!s}")

        return (False, crawler_domain)


def _add_to_metadata(metadata: dict, domain: str, successfull_crawl: bool) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = metadata[domain].get("successful", 0) + successfull_crawl
    metadata[domain]["total"] = metadata[domain].get("total", 0) + 1

    return metadata
