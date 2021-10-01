from Main import *
from flask import Flask, render_template,request
from time import sleep
import os

path11="Name-entity.html"
path22="Dependency_Pattern.html"
path33="sentiment.html"

domain=""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("Sample2.html")




@app.route('/next_page', methods=['GET', 'POST'])
def next():
    ch = request.form.get('Domain_list')
    if ch=="all":
        domain=""
    elif ch=="India":
        domain="national"
    else:
        domain=ch.lower()
        
    runall(domain)
    sleep(30)
    dval=domain.upper()
    return render_template("Sample3.html",domain_list=dval,p1=path11,p2=path22,p3=path33)


@app.route("/Name-entity.html")
def name_entity():
    return render_template("Name-entity.html")

@app.route("/Dependency_Pattern.html")
def dependency_pattern():
    return render_template("Dependency_Pattern.html")

@app.route("/sentiment.html")
def sentiment_():
    return render_template("sentiment.html")

@app.route("/Sample2.html")
def home():
    return render_template("Sample2.html")



if __name__ == "__main__":
    app.run(debug=False)



