import os


class Config:
    SECRET_KEY = os.environ.get('PROD_SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('PROD_SQLALCHEMY_DATABASE_URI')
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get('EMAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('EMAIL_PASS')
