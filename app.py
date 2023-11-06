from flask import Flask
from flask_restful import Resource, Api, reqparse
import pickle
import re




app = Flask(__name__)
api = Api(app)

with open("resume_category_model/tfidf", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)

with open("resume_category_model/clf", "rb") as clf_file:
    clf = pickle.load(clf_file)

category_mapping = {
    6: "Data Science",
    12: "HR",
    0: "Advocate",
    1: "Arts",
    24: "Web Designing",
    16: "Mechanical Engineer",
    22: "Sales",
    14: "Health and fitness",
    5: "Civil Engineer",
    15: "Java Developer",
    4: "Business Analyst",
    21: "SAP Developer",
    2: "Automation Testing",
    11: "Electrical Engineering",
    18: "Operations Manager",
    20: "Python Developer",
    8: "DevOps Engineer",
    17: "Network Security Engineer",
    19: "PMO",
    7: "Database",
    13: "Hadoop",
    10: "ETL Developer",
    9: "DotNet Developer",
    3: "Blockchain",
    23: "Testing",
}


def CleanResume(resume_text):
    cleanText = re.sub("http\S+\s", " ", resume_text)
    cleanText = re.sub("RT|cc", " ", cleanText)
    cleanText = re.sub("#\S+\s", " ", cleanText)
    cleanText = re.sub("@\S+", "  ", cleanText)
    cleanText = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleanText
    )
    cleanText = re.sub(r"[^\x00-\x7f]", " ", cleanText)
    cleanText = re.sub("\s+", " ", cleanText)
    return cleanText


def predict_category(resume_text):
    cleaned_resume = CleanResume(resume_text)
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    category_name = category_mapping.get(prediction_id, "Unknown")
    return category_name


class PredictCategory(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resume_text', type=str, required=True, help='Resume text is required.')
        args = parser.parse_args()

        resume_text = args['resume_text']
        category = predict_category(resume_text)

        return {'category': category}

api.add_resource(PredictCategory, '/predict_category')

if __name__ == '__main__':
    app.run(debug=True)

