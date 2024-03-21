DROP TABLE IF EXISTS anaart;
DROP TABLE IF EXISTS sottofamiglie_comm;
DROP TABLE IF EXISTS famiglie_comm;
DROP TABLE IF EXISTS settori_comm;
DROP TABLE IF EXISTS linee_comm;
DROP TABLE IF EXISTS anacli;
DROP TABLE IF EXISTS tabprov;
DROP TABLE IF EXISTS ute;

CREATE TABLE ute(
    nom_ute varchar(64) NOT NULL,
    cog_ute varchar(64) NOT NULL,
    dat_ute date NOT NULL,
    mai_ute varchar(64) NOT NULL UNIQUE,
    use_ute varchar(64),
    pas_ute varchar(64) NOT NULL,
    amm_ute boolean DEFAULT FALSE,
    PRIMARY KEY(use_ute)
);

CREATE TABLE tabprov(
    cod_prov varchar(2),
    des_prov varchar(40),
    PRIMARY KEY(cod_prov)
);

CREATE TABLE anacli(
    cod_cli int,
    rag_soc varchar(80),
    cod_prov varchar(2),
    PRIMARY KEY(cod_cli),
    FOREIGN KEY(cod_prov) REFERENCES tabprov(cod_prov) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE linee_comm(
    cod_linea_comm varchar(2),
    linea_comm varchar(40),
    PRIMARY KEY(cod_linea_comm)
);

CREATE TABLE settori_comm(
    cod_sett_comm varchar(2),
    sett_comm varchar(40),
    PRIMARY KEY(cod_sett_comm)
);

CREATE TABLE famiglie_comm(
    cod_fam_comm varchar(2),
    fam_comm varchar(40),
    PRIMARY KEY(cod_fam_comm)
);

CREATE TABLE sottofamiglie_comm(
    cod_sott_comm varchar(2),
    sott_comm varchar(40),
    PRIMARY KEY(cod_sott_comm)
);

CREATE TABLE anaart(
    cod_art varchar(13),
    des_art varchar(255),
    cod_linea_comm varchar(2),
    cod_sett_comm varchar(2),
    cod_fam_comm varchar(2),
    cod_sott_comm varchar(2),
    PRIMARY KEY(cod_art),
    FOREIGN KEY(cod_linea_comm) REFERENCES linee_comm(cod_linea_comm) ON UPDATE CASCADE ON DELETE SET NULL,
    FOREIGN KEY(cod_sett_comm) REFERENCES settori_comm(cod_sett_comm) ON UPDATE CASCADE ON DELETE SET NULL,
    FOREIGN KEY(cod_fam_comm) REFERENCES famiglie_comm(cod_fam_comm) ON UPDATE CASCADE ON DELETE SET NULL,
    FOREIGN KEY(cod_sott_comm) REFERENCES sottofamiglie_comm(cod_sott_comm) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE ordclidet(
    cod_cli int,
    cod_art varchar(13),
    data_ord date,
    qta_ordinata float,
    PRIMARY KEY(cod_cli, cod_art, data_ord, qta_ordinata),
    FOREIGN KEY(cod_cli) REFERENCES anacli(cod_cli) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY(cod_art) REFERENCES anaart(cod_art) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE ordclidet_feedback(
    cod_cli int,
    cod_art varchar(13),
    data_ord date,
    qta_ordinata float,
    PRIMARY KEY(cod_cli, cod_art, data_ord, qta_ordinata),
    FOREIGN KEY(cod_cli) REFERENCES anacli(cod_cli) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY(cod_art) REFERENCES anaart(cod_art) ON UPDATE CASCADE ON DELETE CASCADE
);

INSERT INTO ute VALUES 
("Mario", "Rossi", "1994-09-10", "mario.rossi@ergon.it", "a", "a", TRUE),
("Luca", "Verdi", "1994-09-10", "luca.verdi@ergon.it", "b", "b", FALSE);

LOAD DATA INFILE '/dataset/tabprov.csv'
INTO TABLE tabprov
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/dataset/anacli.csv'
INTO TABLE anacli
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/dataset/settori_comm.csv'
INTO TABLE settori_comm
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/dataset/famiglie_comm.csv'
INTO TABLE famiglie_comm
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/dataset/linee_comm.csv'
INTO TABLE linee_comm
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/dataset/sottofamiglie_comm.csv'
INTO TABLE sottofamiglie_comm
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/dataset/anaart.csv'
INTO TABLE anaart
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/dataset/ordclidet.csv'
INTO TABLE ordclidet
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(cod_cli, cod_art, @data_ord, qta_ordinata)
SET data_ord = STR_TO_DATE(@data_ord, '%d/%m/%Y');

LOAD DATA INFILE '/dataset/ordclidet_feedback.csv'
INTO TABLE ordclidet_feedback
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(cod_cli, cod_art, @data_ord, qta_ordinata)
SET data_ord = STR_TO_DATE(@data_ord, '%d/%m/%Y');



