from flask import Flask, render_template, request
import pickle

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()
    

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def hello_world():
    if request.method=="POST":
        myDict = request.form
        fever = int(myDict['fever'])
        bodyPain = int(myDict['bodyPain'])
        diffBreath = int(myDict['diffBreath'])
        RunnyNose = int(myDict['RunnyNose'])
        Age = int(myDict['Age'])


        
        # Code for inference
        inputFeatures = [fever, bodyPain, Age, RunnyNose, diffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
        # return "Hello, World!" + str(infProb) 


if __name__=='__main__':
    app.run(debug=True)