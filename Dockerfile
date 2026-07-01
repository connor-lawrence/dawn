FROM python:3.12
RUN pip install --no-cache-dir numpy
WORKDIR /home/dawn
CMD ["bash"]