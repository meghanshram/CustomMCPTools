from mcp.server.fastmcp import FastMCP
import smtplib
from email.mime.text import MIMEText
from typing import Optional
import os

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "EmailService",  # Name of the MCP server
    instructions="You are an email assistant that can send emails to customers with a predefined SMTP configuration.",  # Instructions for the LLM
    host="0.0.0.0",  # Host address
    port=8006,  # Port number
)

@mcp.tool()
async def send_email(to_email: str, subject: str, body: str) -> str:
    """
    Send an email to a specified recipient.

    This function sends an email using a predefined SMTP server (e.g., Gmail).
    Requires SMTP_USER and SMTP_PASSWORD environment variables to be set.

    Args:
        to_email (str): The recipient's email address.
        subject (str): The subject line of the email.
        body (str): The content of the email.

    Returns:
        str: A message indicating success or failure of the email sending process.
    """
    try:
        # SMTP configuration from environment variables
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if not smtp_user or not smtp_password:
            raise ValueError("SMTP_USER and SMTP_PASSWORD environment variables are required")

        # Create the email message
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        # Connect to SMTP server and send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return f"Email successfully sent to {to_email} with subject: {subject}"
    except Exception as e:
        return f"Error sending email: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
