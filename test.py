from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/')
def hello_name(name):
    return "Hello!"

if __name__ == '__main__':
    app.run()