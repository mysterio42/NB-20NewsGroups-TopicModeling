from flask import Flask, request, jsonify
from flask_restful import Api, Resource

from utils.embedding import load_embedding, latest_modified_embedding
from utils.model import load_model, latest_modified_weight, predict_model

app = Flask(__name__)
api = Api(app)

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
categories.sort()

embedding = load_embedding(latest_modified_embedding())
model = load_model(latest_modified_weight())


class TopicModeling(Resource):

    def post(self):
        posted_data = request.get_json()

        assert 'sentence' in posted_data

        sentence = list(posted_data.values())
        embed_sentence = embedding.embed_unseen(sentence)
        topic = predict_model(model, embed_sentence, categories)
        return jsonify({'prediction': {'topic': topic}})


api.add_resource(TopicModeling, '/topic')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
