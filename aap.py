/p>"+ val
from flask import Flask

app = Flask(__name__)

@app.route("/hello/<val>")
def hello_world():
    return "<p>Hello, World!<
@app.route("/sum/<x>/<y>")
def sum_num(x,y):
    sum= int(x) + int(y)
    return str(sum)