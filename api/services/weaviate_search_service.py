import json
import logging
import time
from typing import List, Dict, Any

import numpy as np
from flask import current_app
from langchain import requests
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import weaviate
from sklearn.manifold import TSNE

from core.embedding.cached_embedding import CacheEmbedding
from core.index.vector_index.vector_index import VectorIndex
from core.model_providers.model_factory import ModelFactory
from extensions.ext_database import db
from models.account import Account
from models.dataset import Dataset, DocumentSegment, DatasetQuery


class WeaviateService:

    def __init__(self, config: dict):
        self._client = self._init_client(config)

    def _init_client(self, config: dict):
        auth_config = weaviate.auth.AuthApiKey(api_key=config.api_key)
        weaviate.connect.connection.has_grpc = False
        try:
            client = weaviate.Client(
                url=config.endpoint,
                auth_client_secret=auth_config,
                timeout_config=(5, 60),
                startup_period=None,
                additional_headers = {
                "X-OpenAI-Api-Key": current_app.config.get('WEAVIATE_ENDPOINT'),  # Replace with your inference API key
                 }
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError('Vector database connection error')
        return client


    def save_dataset_query(self, dataset_id, query, account: Account):
        dataset_query = DatasetQuery(
            dataset_id=dataset_id,
            content=query,
            source='hit_testing',
            created_by_role='account',
            created_by=account.id
        )

        db.session.add(dataset_query)
        db.session.commit()

    @classmethod
    def retrieve(cls, dataset: Dataset, query: str, where_condition: dict, limit: int = 10) -> dict:
        if dataset.available_document_count == 0 or dataset.available_segment_count == 0:
            return {
                "query": {
                    "content": query,
                    "tsne_position": {'x': 0, 'y': 0},
                },
                "records": []
            }

        embedding_model = ModelFactory.get_embedding_model(
            tenant_id=dataset.tenant_id,
            model_provider_name=dataset.embedding_model_provider,
            model_name=dataset.embedding_model
        )

        embeddings = CacheEmbedding(embedding_model)

        vector_index = VectorIndex(
            dataset=dataset,
            config=current_app.config,
            embeddings=embeddings
        )
        client = vector_index._vector_index._client
        dataset_class="Vector_index_" + dataset.id.replace("-", "_") + '_Node'
        start = time.perf_counter()
        query_obj = client.query.get(dataset_class, ['text', 'doc_id', 'dataset_id', 'document_id'])
        embedded_query = embeddings.embed_query(query)
        vector = {"vector": embedded_query}
        if where_condition is not None:
            result = (
                query_obj.with_near_vector(vector)
                .with_limit(limit)
                .with_additional("vector")
                .with_where(where_condition)
                .do()
             )
        else:
            result = (
                query_obj.with_near_vector(vector)
                .with_limit(limit)
                .with_additional("vector")
                .do()
            )

        docs_and_scores = []
        for res in result["data"]["Get"][client._index_name]:
            text = res.pop(client._text_key)
            score = np.dot(res["_additional"]["vector"], embedded_query)
            docs_and_scores.append((Document(page_content=text, metadata=res), score))


        docs = []
        for doc, similarity in docs_and_scores:
            doc.metadata['score'] = similarity
            docs.append(doc)

        # documents = vector_index.search(
        #     query,
        #     search_type='similarity_score_threshold',
        #     search_kwargs={
        #         'k': 10,
        #         'filter': {
        #             'group_id': [dataset.id]
        #         }
        #     }
        # )
        end = time.perf_counter()
        logging.debug(f"Weaviate Search Service in {end - start:0.4f} seconds")

        # dataset_query = DatasetQuery(
        #     dataset_id=dataset.id,
        #     content=query,
        #     source='hit_testing',
        #     created_by_role='account',
        #     created_by=account.id
        # )
        #
        # db.session.add(dataset_query)
        # db.session.commit()

        return cls.compact_retrieve_response(dataset, embeddings, query, docs)

    @classmethod
    def compact_retrieve_response(cls, dataset: Dataset, embeddings: Embeddings, query: str, documents: List[Document]):
        text_embeddings = [
            embeddings.embed_query(query)
        ]

        text_embeddings.extend(embeddings.embed_documents([document.page_content for document in documents]))

        tsne_position_data = cls.get_tsne_positions_from_embeddings(text_embeddings)

        query_position = tsne_position_data.pop(0)

        i = 0
        records = []
        for document in documents:
            index_node_id = document.metadata['doc_id']

            segment = db.session.query(DocumentSegment).filter(
                DocumentSegment.dataset_id == dataset.id,
                DocumentSegment.enabled == True,
                DocumentSegment.status == 'completed',
                DocumentSegment.index_node_id == index_node_id
            ).first()

            if not segment:
                i += 1
                continue

            record = {
                "segment": segment,
                "score": document.metadata['score'],
                "tsne_position": tsne_position_data[i]
            }

            records.append(record)

            i += 1

        return {
            "query": {
                "content": query,
                "tsne_position": query_position,
            },
            "records": records
        }

    @classmethod
    def get_tsne_positions_from_embeddings(cls, embeddings: list):
        embedding_length = len(embeddings)
        if embedding_length <= 1:
            return [{'x': 0, 'y': 0}]

        concatenate_data = np.array(embeddings).reshape(embedding_length, -1)
        # concatenate_data = np.concatenate(embeddings)

        perplexity = embedding_length / 2 + 1
        if perplexity >= embedding_length:
            perplexity = max(embedding_length - 1, 1)

        tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12.0)
        data_tsne = tsne.fit_transform(concatenate_data)

        tsne_position_data = []
        for i in range(len(data_tsne)):
            tsne_position_data.append({'x': float(data_tsne[i][0]), 'y': float(data_tsne[i][1])})

        return tsne_position_data



    def single_import_data(self,class_name: str, article_id:str,summary:str,keypoint:str,user_id:str):
        uuid = self._client.data_object.create(
            data_object={
                'article_id': article_id,
                'summary': summary,
                'keypoint': keypoint,
                'user_id': user_id,
            },
            class_name=class_name,
        )
        return uuid

    def batch_import_data(self,class_name: str, data: json):
        self._client.batch.configure(batch_size=100)  # Configure batch
        with self._client.batch as batch:  # Initialize a batch process
            for i, d in enumerate(data):  # Batch import data
                print(f"importing question: {i+1}")
                properties = {
                    "article_id": d["Article_id"],
                    "summary": d["Summary"],
                    "keypoint": d["Keypoints"],
                    "user_id": d["UserId"],
                }
                batch.add_data_object(
                    data_object=properties,
                    class_name=class_name
                )

    # ===== import data =====

    #### query ####
    def search(self,classname:str,query:str,user_id:str,limit:10):
        response = (
            self._client.query
            .get(classname, ["article_id", "summary", "keypoint","user_id"])
            .with_hybrid(
                query=query
            )
            .with_where({
                "path": ["user_id"],
                "operator": "Equal",
                "valueText": user_id
            })
            .with_additional("score")
            .with_limit(limit)
            .do()
        )
        print(json.dumps(response, indent=4, ensure_ascii=False))
        return response
