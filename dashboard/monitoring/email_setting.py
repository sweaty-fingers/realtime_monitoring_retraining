import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

email_settings = {
        'sender': 'swbootcampt8@naver.com',
        'receiver': 'swbootcampt8@naver.com',
        'password': 'swbootcampt*',
        'smtp_server': 'smtp.naver.com',
        'smtp_port': 587
    }

class EmailRealTime():
    def send_email_alert(self, model_name, accuracy, placeholder):
        if email_settings:
            sender_email = email_settings['sender']
            receiver_email = email_settings['receiver']
            password = email_settings['password']
            smtp_server = email_settings['smtp_server']
            smtp_port = email_settings['smtp_port']

            subject = f'Alert: {model_name} Accuracy Below Threshold'
            body = f'The accuracy of {model_name} has fallen below the threshold. Current accuracy: {accuracy:.2f}. Start retraining'

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            try:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, password)
                    server.send_message(msg)
                self.email_sent = True  # 이메일이 발송되었음을 표시
                placeholder.write(f"Email alert sent to {receiver_email} for {model_name} accuracy below threshold. Start retraining")
            except Exception as e:
                placeholder.write(f"Failed to send email: {e}")


class EmailRealLoss():
    def send_email_alert(self, model_name, loss, epoch, placeholder):
        if email_settings:
            sender_email = email_settings['sender']
            receiver_email = email_settings['receiver']
            password = email_settings['password']
            smtp_server = email_settings['smtp_server']
            smtp_port = email_settings['smtp_port']

            subject = f'Alert: {model_name} Training Completed'
            body = f'Training {model_name} Completed. ACC/Epoch: {loss:.2f}/{epoch}'

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            try:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, password)
                    server.send_message(msg)
                self.email_sent = True  # 이메일이 발송되었음을 표시
                placeholder.write(f"Email alert sent to {receiver_email} for {model_name} training comeplete")
            except Exception as e:
                placeholder.write(f"Failed to send email: {e}")


class EmailRealAB():
    def send_email_alert(self, model_name, placeholder):
        if email_settings:
            sender_email = email_settings['sender']
            receiver_email = email_settings['receiver']
            password = email_settings['password']
            smtp_server = email_settings['smtp_server']
            smtp_port = email_settings['smtp_port']

            subject = f'Alert: {model_name} Better model trained'
            body = f'{model_name} performance checked.'

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            try:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, password)
                    server.send_message(msg)
                self.email_sent = True  # 이메일이 발송되었음을 표시
                placeholder.write(f"Email alert sent to {receiver_email} for f'{model_name} performance checked.")
            except Exception as e:
                placeholder.write(f"Failed to send email: {e}")