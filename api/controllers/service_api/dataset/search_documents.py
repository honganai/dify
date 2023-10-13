import logging

from flask_login import current_user

from controllers.service_api.wraps import DatasetApiResource
from core.index.vector_index.weaviate_vector_index import WeaviateConfig
from core.login.login import login_required
from flask_restful import Resource, reqparse, marshal
from werkzeug.exceptions import InternalServerError, NotFound, Forbidden

import services
from controllers.service_api import api
from controllers.console.app.error import ProviderNotInitializeError, ProviderQuotaExceededError, \
    ProviderModelCurrentlyNotSupportError
from controllers.console.datasets.error import HighQualityDatasetOnlyError, DatasetNotInitializedError
from controllers.console.setup import setup_required
from controllers.console.wraps import account_initialization_required
from core.model_providers.error import ProviderTokenNotInitError, QuotaExceededError, ModelCurrentlyNotSupportError, \
    LLMBadRequestError
from fields.hit_testing_fields import hit_testing_record_fields
from services.dataset_service import DatasetService
from services.hit_testing_service import HitTestingService
from services.weaviate_search_service import WeaviateService
from flask import current_app, json


class SearchDocumentApi(DatasetApiResource):
    def post(self, tenant_id, dataset_id):
        dataset_id_str = str(dataset_id)

        dataset = DatasetService.get_dataset(dataset_id_str)
        if dataset is None:
            raise NotFound("Dataset not found.")

        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))

        # only high quality dataset can be used for hit testing
        if dataset.indexing_technique != 'high_quality':
            raise HighQualityDatasetOnlyError()

        parser = reqparse.RequestParser()
        parser.add_argument('query', type=str, location='json')
        parser.add_argument('user_id', type=str, location='json')
        args = parser.parse_args()

        query = args['query']
        user_id = args['user_id']
        where_condition = None
        if not query or len(query) > 250:
            raise ValueError('query is required and cannot exceed 250 characters')
        if user_id and len(user_id) > 1:
            #where_condition转为dict
            where_condition = {
                "path": ["keywords"],
                "operator": "Equal",
                "valueText": user_id
            }

        try:
            response = WeaviateService.retrieve(
                dataset=dataset,
                query=query,
                where_condition=where_condition,
                limit=10,
            )

            return {"query": response['query'], 'records': marshal(response['records'], hit_testing_record_fields)}
        except services.errors.index.IndexNotInitializedError:
            raise DatasetNotInitializedError()
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        except LLMBadRequestError:
            raise ProviderNotInitializeError(
                f"No Embedding Model available. Please configure a valid provider "
                f"in the Settings -> Model Provider.")
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            logging.exception("SearchDocumentApi  failed.")
            raise InternalServerError(str(e))


class CreateSegmentApi(DatasetApiResource):
    def post(self, tenant_id):
        config=WeaviateConfig(
            endpoint=current_app.config.get('WEAVIATE_ENDPOINT'),
            api_key=current_app.config.get('WEAVIATE_API_KEY'),
            batch_size=int(current_app.config.get('WEAVIATE_BATCH_SIZE'))
        )
        weaviate=WeaviateService(config)
        parser = reqparse.RequestParser()
        parser.add_argument('article_id', type=str, required=True, nullable=True, location='json')
        parser.add_argument('summary', type=str, required=True, nullable=True, location='json')
        parser.add_argument('keypoints', type=str, required=True, nullable=False, location='json')
        parser.add_argument('user_id', type=str, default='', required=True, nullable=False, location='json')
        parser.add_argument('class_name', type=str, default='dataset_keypoints_all_user', required=False, nullable=True,
                            location='json')
        args = parser.parse_args()
        article_id = args['article_id']
        summary = args['summary']
        keypoint = args['keypoints']
        user_id = args['user_id']
        class_name = args['class_name']

        segment_uuid = weaviate.single_import_data(class_name, article_id, summary, keypoint, user_id)
        return {'segment_uuid':segment_uuid},200


class BatchCreateSegmentApi(DatasetApiResource):
    def post(self, tenant_id):
        config=WeaviateConfig(
            endpoint=current_app.config.get('WEAVIATE_ENDPOINT'),
            api_key=current_app.config.get('WEAVIATE_API_KEY'),
            batch_size=int(current_app.config.get('WEAVIATE_BATCH_SIZE'))
        )
        weaviate=WeaviateService(config)
        parser = reqparse.RequestParser()
        parser.add_argument('data', type=json, required=True, nullable=True, location='json')
        parser.add_argument('class_name', type=str, default='dataset_keypoints_all_user', required=False, nullable=True,
                            location='json')
        args = parser.parse_args()
        data = args['data']
        class_name = args['class_name']
        try:
            weaviate.batch_import_data(class_name, data)
        except Exception as e:
            logging.exception("BatchCreateSegmentApi failed.")
            return "batch uploading failed", 500
        return {'message':"batch uploading"}, 200


class SearchSegmentApi(DatasetApiResource):
    def post(self, tenant_id):
        config=WeaviateConfig(
            endpoint=current_app.config.get('WEAVIATE_ENDPOINT'),
            api_key=current_app.config.get('WEAVIATE_API_KEY'),
            batch_size=int(current_app.config.get('WEAVIATE_BATCH_SIZE'))
        )
        warviate=WeaviateService(config)
        parser = reqparse.RequestParser()
        parser.add_argument('query', type=str, required=True, nullable=False, location='json')
        parser.add_argument('Limit', type=int, default=10, required=False, nullable=False, location='json')
        parser.add_argument('user_id', type=str, default='', required=True, nullable=False, location='json')
        parser.add_argument('class_name', type=str, default='dataset_keypoints_all_user', required=False, nullable=True,
                            location='json')
        args = parser.parse_args()
        query = args['query']
        user_id= args['user_id']
        limit = args['Limit']
        class_name = args['class_name']

        return warviate.search(class_name,query,user_id,limit),200


api.add_resource(BatchCreateSegmentApi, '/datasets/batch_create_segment')
api.add_resource(SearchSegmentApi, '/datasets/search_segment')
api.add_resource(CreateSegmentApi, '/datasets/create_segment')
api.add_resource(SearchDocumentApi, '/datasets/<uuid:dataset_id>/search_document')
