from flask_wtf import FlaskForm
from flask_login import current_user
import wtforms as forms
import wtforms.validators as validators

import real_estate_analysis.app.app.models as models


class RegistrationForm(FlaskForm):
    username = forms.StringField(label='Username', validators=[validators.DataRequired(), validators.Length(min=2, max=20)])
    email = forms.StringField(label='Email', validators=[validators.DataRequired(), validators.Email()])
    password = forms.PasswordField(label='Password', validators=[validators.DataRequired()])
    confirm_password = forms.PasswordField(label='Confirm Password', validators=[validators.DataRequired(), validators.EqualTo('password')])
    submit = forms.SubmitField(label='Sign Up')

    def validate_username(self, username):
        user = models.User.query.filter_by(username=username.data).first()
        if user:
            raise validators.ValidationError('That username is taken. Please choose a different username')

    def validate_email(self, email):
        user = models.User.query.filter_by(email=email.data).first()
        if user:
            raise validators.ValidationError('That email is taken. Please choose a different email')


class LoginForm(FlaskForm):
    email = forms.StringField(label='Email', validators=[validators.DataRequired(), validators.Email()])
    password = forms.PasswordField(label='Password', validators=[validators.DataRequired()])
    remember = forms.BooleanField(label='Remember Me')
    submit = forms.SubmitField(label='Login')


class UpdateAccountForm(FlaskForm):
    username = forms.StringField(label='Username', validators=[validators.DataRequired(), validators.Length(min=2, max=20)])
    email = forms.StringField(label='Email', validators=[validators.DataRequired(), validators.Email()])
    submit = forms.SubmitField(label='Update')

    def validate_username(self, username):
        if username.data != current_user.username:
            user = models.User.query.filter_by(username=username.data).first()
            if user:
                raise validators.ValidationError('That username is taken. Please choose a different username')

    def validate_email(self, email):
        if email.data != current_user.email:
            user = models.User.query.filter_by(email=email.data).first()
            if user:
                raise validators.ValidationError('That email is taken. Please choose a different email')


class RequestResetForm(FlaskForm):
    email = forms.StringField(label='Email', validators=[validators.DataRequired(), validators.Email()])
    submit = forms.SubmitField(label='Request Password Request')

    def validate_email(self, email):
        user = models.User.query.filter_by(email=email.data).first()
        if user is None:
            raise validators.ValidationError('There is no account with the provided email. Please register for an account')


class ResetPasswordForm(FlaskForm):
    password = forms.PasswordField(label='Password', validators=[validators.DataRequired()])
    confirm_password = forms.PasswordField(label='Confirm Password',
                                           validators=[validators.DataRequired(), validators.EqualTo('password')])
    submit = forms.SubmitField(label='Reset Password')
