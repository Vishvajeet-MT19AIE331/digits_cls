from flask import Flask

app = Flask(__name__)
'''
@app.route("/")
def hello_world():
    return "<p>Hello, World!<p>"
'''    
@app.route("/sum/<x>/<y>")
#@app.route("/<x>/<y>")
def sum_num(x,y):
    sum= int(x) + int(y)
    return str(sum)