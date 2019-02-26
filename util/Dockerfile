FROM python:3.5-slim-jessie
# add tati package and its dependencies
RUN pip install tati jupyter
RUN mkdir -p /home/jovyan/tati
WORKDIR /home/jovyan/tati
COPY guided-tour-*.ipynb ./
COPY dataset*.csv ./
RUN ls -l /home/jovyan/tati

# https://jupyter-notebook.readthedocs.io/en/stable/public_server.html
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

