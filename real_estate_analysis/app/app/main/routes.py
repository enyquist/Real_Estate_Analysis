from flask import render_template, Blueprint

main = Blueprint('main', __name__)

posts = [
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second Post Content',
        'date_posted': 'April 21, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 1',
        'content': 'First Post Content',
        'date_posted': 'April 20, 2018'
    }
]


@main.route('/')
@main.route('/home')
def home():
    return render_template('home.html')


@main.route('/about')
def about():
    return render_template('about.html', title='About')


@main.route('/faq')
def faq():
    return render_template('faq.html', title='FAQ')


@main.route('/pricing')
def pricing():
    return render_template('pricing.html', title='Pricing')


@main.route('/contact_us')
def contact_us():
    return render_template('contact_us.html', title='Contact Us')
