from flask import Blueprint, render_template

blueprint = Blueprint("blueprint", __name__, static_folder="static", template_folder="templates")

@blueprint.route("/home")
@blueprint.route("/")
def home():
  return "<h1>Hello!</h1>"
  # return render_template("homepage.html")