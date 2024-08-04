@observe(as_type="generation")
def call_openai(
    llm_params,
    system_text,
    text,
    logger,
    log_prefix="",
    history=None,
    temperature=0.3,
    model="openai",
    criteria=None,
    stage="",
):
    out = ""
    cached_result = ""
    json_data = {
        "messages": [
            {
                "role": "system",
                "content": f"{system_text}",
            },
            {"role": "user", "content": f"{text}"},
        ],
        "temperature": 0.1,
        "seed": 42,
    }
    if history:
        json_data["messages"] += history
    cache_md5_json = json_data
    cache_md5_json["model"] = model
    cache_key = hashlib.md5(str(cache_md5_json).encode("utf-8")).hexdigest()
    # Check if the result is already cached in Redis
    if llm_params["ENABLE_REDIS_CACHE"]:
        try:
            with Redis(connection_pool=redis_pool) as redis_client:
                cached_result = redis_client.get(cache_key)
        except Exception as e:
            logger.error(f"Error retrieving result from cache LLM call: {e}")
            cached_result = ""
    if cached_result:
        logger.info(f"Retrieving result from cache for {cache_key}")
        out = cached_result.decode("utf-8")
        time.sleep(random.uniform(3.2, 3.5))  # Add a random delay to log langfuse
    if out == "":
        active_llm = get_active_llm(
            available_llms=llm_params, active_llm=model, logger=logger
        )
        out = active_llm.predict(
            system_text,
            text,
            history=history,
            temperature=temperature,
            criteria=criteria,
            log_prefix=log_prefix,
        )
        # Cache the result in Redis
        try:
            with Redis(connection_pool=redis_pool) as redis_client:
                redis_client.set(cache_key, out)
        except Exception as e:
            logger.error(f"Error caching result in cache LLM call: {e}")
    # out = call_openai_wrap(llm_params, system_text, text, logger, log_prefix = log_prefix, history = history,
    #                        temperature = temperature,
    #                        model = model)
    log_prefix["stage"] = stage
    log_prefix["model"] = model
    log_prefix["cache_hash"] = cache_key
    log_prefix["cached"] = bool(cached_result)
    langfuse_context.update_current_observation(
        model="gpt-4-1106-preview" if model == "openai" else model,
        input=json_data,
        output=out,
        metadata=log_prefix,
        name=stage,
    )
    json_data["out"] = out
    logger.info(f"Response from {model} for {log_prefix}, with : {json_data}")
    return out
