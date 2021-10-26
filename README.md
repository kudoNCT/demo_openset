# Demo Face recogniton Website
# Run step by step
- sudo chmod 666 /var/run/docker.sock
- heroku git:remote -a <App_name>
- heroku container:login
- heroku container:push web
- heroku container:release web
- heroku open

