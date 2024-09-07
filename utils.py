def handle_streaming_json(streaming_json):    
    for chunk in streaming_json:
        if 'answer' in chunk:
            yield chunk['answer'] 