# -*- coding:utf-8 -*-
import json
import logging
from werkzeug.exceptions import NotFound
import flask_restful
from flask import request
# from flask_login import current_user
from flask_restful import fields, marshal_with, reqparse, inputs, abort, Resource

from constants.model_template import model_templates
from controllers.console.apikey import _get_resource
from controllers.console.app import _get_app
from controllers.service_api import api
from controllers.service_api.app.error import ProviderNotInitializeError
from controllers.service_api.wraps import AppApiResource
from core.model_providers.error import ProviderTokenNotInitError, LLMBadRequestError
from core.model_providers.model_factory import ModelFactory
from core.model_providers.model_provider_factory import ModelProviderFactory
from events.app_event import app_was_created, app_model_config_was_updated
from extensions.ext_database import db
from libs.helper import TimestampField
from models.account import Account

from models.model import App, AppModelConfig, Site, ApiToken
from services.app_model_config_service import AppModelConfigService


api_key_fields = {
    'id': fields.String,
    'type': fields.String,
    'token': fields.String,
    'last_used_at': TimestampField,
    'created_at': TimestampField
}

api_key_list = {
    'data': fields.List(fields.Nested(api_key_fields), attribute="items")
}

current_tenant_id = "eabcc971-b1af-4844-8c43-0c2d2229339b"

class AppParameterApi(AppApiResource):
    """Resource for app variables."""

    variable_fields = {
        'key': fields.String,
        'name': fields.String,
        'description': fields.String,
        'type': fields.String,
        'default': fields.String,
        'max_length': fields.Integer,
        'options': fields.List(fields.String)
    }

    parameters_fields = {
        'opening_statement': fields.String,
        'suggested_questions': fields.Raw,
        'suggested_questions_after_answer': fields.Raw,
        'speech_to_text': fields.Raw,
        'retriever_resource': fields.Raw,
        'more_like_this': fields.Raw,
        'user_input_form': fields.Raw,
    }

    @marshal_with(parameters_fields)
    def get(self, app_model: App, end_user):
        """Retrieve app parameters."""
        app_model_config = app_model.app_model_config

        return {
            'opening_statement': app_model_config.opening_statement,
            'suggested_questions': app_model_config.suggested_questions_list,
            'suggested_questions_after_answer': app_model_config.suggested_questions_after_answer_dict,
            'speech_to_text': app_model_config.speech_to_text_dict,
            'retriever_resource': app_model_config.retriever_resource_dict,
            'more_like_this': app_model_config.more_like_this_dict,
            'user_input_form': app_model_config.user_input_form_list
        }


class AppListApi(Resource):

    def get(self):
        """Get app list"""
        parser = reqparse.RequestParser()
        parser.add_argument('page', type=inputs.int_range(1, 99999), required=False, default=1, location='args')
        parser.add_argument('limit', type=inputs.int_range(1, 100), required=False, default=20, location='args')
        args = parser.parse_args()

        app_models = db.paginate(
            db.select(App).where(App.tenant_id == current_tenant_id,
                                 App.is_universal == False).order_by(App.created_at.desc()),
            page=args['page'],
            per_page=args['limit'],
            error_out=False)

        return {
            'total': app_models.total,
            'page': app_models.page,
            'limit': app_models.per_page,
            'items': [{
                'id': app_model.id,
                'name': app_model.name,
                'mode': app_model.mode,
                'icon': app_model.icon,
                'icon_background': app_model.icon_background,
            } for app_model in app_models.items]
        }


    def post(self):
        """Create app"""
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str, required=True, location='json')
        parser.add_argument('mode', type=str, choices=['completion', 'chat'], location='json')
        parser.add_argument('icon', type=str, location='json')
        parser.add_argument('icon_background', type=str, location='json')
        parser.add_argument('model_config', type=dict, location='json')
        args = parser.parse_args()
        print(args)

        try:
            default_model = ModelFactory.get_text_generation_model(
                tenant_id=current_tenant_id
            )
        except (ProviderTokenNotInitError, LLMBadRequestError):
            default_model = None
        except Exception as e:
            logging.exception(e)
            default_model = None

        if args['model_config'] is not None:
            # validate config
            model_config_dict = args['model_config']

            # get model provider
            model_provider = ModelProviderFactory.get_preferred_model_provider(
                current_tenant_id,
                model_config_dict["model"]["provider"]
            )

            if not model_provider:
                if not default_model:
                    raise ProviderNotInitializeError(
                        f"No Default System Reasoning Model available. Please configure "
                        f"in the Settings -> Model Provider.")
                else:
                    model_config_dict["model"]["provider"] = default_model.model_provider.provider_name
                    model_config_dict["model"]["name"] = default_model.name
            current_user = db.session.query(Account).first()
            current_user.current_tenant_id = current_tenant_id
            model_configuration = AppModelConfigService.validate_configuration(
                tenant_id=current_tenant_id,
                account=current_user,
                config=model_config_dict,
                mode=args['mode']
            )

            app = App(
                enable_site=True,
                enable_api=True,
                is_demo=False,
                api_rpm=0,
                api_rph=0,
                status='normal'
            )

            app_model_config = AppModelConfig()
            app_model_config = app_model_config.from_model_config_dict(model_configuration)
        else:
            if 'mode' not in args or args['mode'] is None:
                abort(400, message="mode is required")

            model_config_template = model_templates[args['mode'] + '_default']

            app = App(**model_config_template['app'])
            app_model_config = AppModelConfig(**model_config_template['model_config'])

            # get model provider
            model_provider = ModelProviderFactory.get_preferred_model_provider(
                current_tenant_id,
                app_model_config.model_dict["provider"]
            )

            if not model_provider:
                if not default_model:
                    raise ProviderNotInitializeError(
                        f"No Default System Reasoning Model available. Please configure "
                        f"in the Settings -> Model Provider.")
                else:
                    model_dict = app_model_config.model_dict
                    model_dict['provider'] = default_model.model_provider.provider_name
                    model_dict['name'] = default_model.name
                    app_model_config.model = json.dumps(model_dict)

        app.name = args['name']
        app.mode = args['mode']
        app.icon = args['icon']
        app.icon_background = args['icon_background']
        app.tenant_id = current_tenant_id

        db.session.add(app)
        db.session.flush()

        app_model_config.app_id = app.id
        db.session.add(app_model_config)
        db.session.flush()

        app.app_model_config_id = app_model_config.id

        site = Site(
            app_id=app.id,
            title=app.name,
            default_language="zh",
            customize_token_strategy='not_allow',
            code=Site.generate_code(16)
        )

        db.session.add(site)
        db.session.commit()

        app_was_created.send(app)

        return {
            'id': app.id,
            'name': app.name,
            'mode': app.mode,
            'icon': app.icon,
            'icon_background': app.icon_background,
        }

class BaseApiKeyListResource(Resource):
    resource_type = None
    resource_model = None
    resource_id_field = None
    token_prefix = None
    max_keys = 1000

    @marshal_with(api_key_list)
    def get(self, resource_id):
        resource_id = str(resource_id)
        _get_resource(resource_id, current_tenant_id,
                      self.resource_model)
        keys = db.session.query(ApiToken). \
            filter(ApiToken.type == self.resource_type, getattr(ApiToken, self.resource_id_field) == resource_id). \
            all()
        return {"items": keys}

    @marshal_with(api_key_fields)
    def post(self, resource_id):
        resource_id = str(resource_id)
        _get_resource(resource_id, current_tenant_id,
                      self.resource_model)



        # current_key_count = db.session.query(ApiToken). \
        #     filter(ApiToken.type == self.resource_type, getattr(ApiToken, self.resource_id_field) == resource_id). \
        #     count()

        key = ApiToken.generate_api_key(self.token_prefix, 24)
        api_token = ApiToken()
        setattr(api_token, self.resource_id_field, resource_id)
        api_token.tenant_id = current_tenant_id
        api_token.token = key
        api_token.type = self.resource_type
        db.session.add(api_token)
        db.session.commit()
        return api_token, 201

class AppApiKeyListResource(BaseApiKeyListResource):

    def after_request(self, resp):
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Credentials'] = 'true'
        return resp

    resource_type = 'app'
    resource_model = App
    resource_id_field = 'app_id'
    token_prefix = 'app-'


class ModelConfigResource(Resource):
    def _get_app_by_tenant(app_id, tenant_id,mode=None):
        app = db.session.query(App).filter(
            App.id == app_id,
            App.tenant_id == tenant_id,
            App.status == 'normal'
        ).first()

        if not app:
            raise NotFound("App not found")

        if mode and app.mode != mode:
            raise NotFound("The {} app not found".format(mode))

        return app
    def post(self, app_id):
        """Modify app model config"""
        app_id = str(app_id)

        app_model = self._get_app_by_tenant(app_id,tenant_id=current_tenant_id)
        current_user=Account
        current_user.current_tenant_id = current_tenant_id
        # validate config
        model_configuration = AppModelConfigService.validate_configuration(
            tenant_id=current_tenant_id,
            account=current_user,
            config=request.json,
            mode=app_model.mode
        )

        new_app_model_config = AppModelConfig(
            app_id=app_model.id,
        )
        new_app_model_config = new_app_model_config.from_model_config_dict(model_configuration)

        db.session.add(new_app_model_config)
        db.session.flush()

        app_model.app_model_config_id = new_app_model_config.id
        db.session.commit()

        app_model_config_was_updated.send(
            app_model,
            app_model_config=new_app_model_config
        )

        return {'result': 'success'}


api.add_resource(ModelConfigResource, '/out/<uuid:app_id>/model-config')
api.add_resource(AppParameterApi, '/parameters')
api.add_resource(AppListApi, '/out_apps')
api.add_resource(AppApiKeyListResource, '/out_apps/<uuid:resource_id>/api-keys')
