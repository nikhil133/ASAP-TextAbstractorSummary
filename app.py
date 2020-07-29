from flask import Flask, request, jsonify
from predict import predict
app=Flask(__name__)

@app.route("/app/v1/incoming")
def GetText():
    text=request.args.get('text')
    print("Request for ",text)
    body={
        'success':True,
        'summary':predict(text)
    }
    return jsonify(body)

if __name__=='__main__':
    app.run(debug=True)