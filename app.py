from flask import Flask, request
from predict import predict
app=Flask(__name__)

@app.route("/app/v1/incoming")
def GetText():
    text=request.args.get('text')
    print(text)
    return predict(text)

if __name__=='__main__':
    app.run(debug=True)