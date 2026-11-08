FROM r-base:4.4.3

WORKDIR /app

COPY src/r_code/install_packages.R .
RUN Rscript install_packages.R

COPY src/r_code/main.R ./src/r_code/main.R

CMD ["Rscript", "src/r_code/main.R"]