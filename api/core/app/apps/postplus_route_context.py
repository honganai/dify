"""Attach server-owned message IDs only to the independent PostPlus provider."""


def postplus_model_parameters(provider: str, parameters: dict, app_id: str, message_id: str) -> dict:
    if provider != "yuanma/postplus_vertex/postplus_vertex":
        return parameters
    return {**parameters, "_postplus_app_id": app_id, "_postplus_message_id": message_id}
