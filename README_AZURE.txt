AZURE DEPLOYMENT

1. Deploy all files with this structure:
   app.py
   requirements.txt
   marketing_campaign_DS14.csv
   templates/index.html

2. Azure Portal -> App Service -> Configuration -> General settings.

3. Set Startup Command to:
   gunicorn --bind=0.0.0.0 --timeout 600 app:app

4. Save and restart.

5. Test:
   https://YOUR-APP-NAME.azurewebsites.net/health
   Expected: {"status":"ok"}

6. Then open the root URL.
