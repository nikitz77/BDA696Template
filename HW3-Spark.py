# HW3 - Spark Assignment
# BDA 696 -  Karenina Zaballa

import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from Transformer_100_Day_AVG import R_AVG_100

# Setup Spark

# PERSONAL ERRORS I HAD TO RESEARCH ON:
# from setuptools import command
# process = subprocess.Popen(command, stdout=tempFile, shell=True)
# https://docs.python.org/3/library/subprocess.html


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"  # This may be named differently, i.e. "baseball"
    port = "3306"
    # input login info here
    user = "root"  # pragma: allowlist secret
    password = "x11docker"  # pragma: allowlist secret
    extras = (
        "useUnicode=true&useJDBCCompliantTimezoneShift=true"
        "+&useLegacyDatetimeCode=false&serverTimezone=PST"
    )

    # STANDARD TABLE
    df = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}?{extras}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="batter_counts",
            user=user,  # pragma: allowlist secret
            password=password,  # pragma: allowlist secret
        )
        .load()
    )
    # Make sure the big table shows
    df.show()

    df.createOrReplaceTempView("batters_counts")
    df.persist(StorageLevel.DISK_ONLY)

    print("Making temp table: ba_temp\n")
    df1 = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}?{extras}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="(SELECT game.game_id, batter_counts.batter, \
                    batter_counts.Hit, batter_counts.atBat, game.local_date \
                    FROM batter_counts\
                    JOIN game ON batter_counts.game_id = game.game_id)ba_temp",
            user=user,  # pragma: allowlist secret
            password=password,  # pragma: allowlist secret
        )
        .load()
    )
    # Make sure ba_temp shows
    df1.show()

    df1.createOrReplaceTempView("ba_temp")
    df1.persist(StorageLevel.DISK_ONLY)

    # Paste ba_temp onto itself to set up for calculations
    # This was my previous attempt to make tables work,
    # but there was a casting issue on the date here

    # To explore later, unix_timestamp library: https://spark.apache.org/docs/latest/api
    # /python/pyspark.sql.html#pyspark.sql.functions.unix_timestamp

    # SELECT ba1.batter, ba1.game_id, SUM(ba2.Hit) AS Hit, \
    # SUM(ba2.atBat) AS atBat, ba2.local_date \
    # FROM   ba_temp ba1 \
    # JOIN   ba_temp ba2 \
    # ON ba1.batter = ba2.batter \
    # AND ba2.local_date > DATE_SUB(ba1.local_date, INTERVAL 100 DAY) \
    # AND ba1.local_date > ba2.local_date \
    # GROUP BY ba1.game_id, ba1.batter,ba1.local_date """

    print("Prepping tables for 100 Day Rolling Average\n")

    tableprep = spark.sql(
        """(SELECT SUM(ba2.Hit) AS TotalHits, SUM(ba2.atBat) AS TotalAtBat, \
            ba1.batter, ba1.game_id \
            FROM   ba_temp ba1 \
            JOIN   ba_temp ba2 \
            ON ba2.batter = ba1.batter AND \
            ba2.local_date BETWEEN DATE_SUB(ba1.local_date,100) AND \
            DATE_SUB(ba1.local_date, 1)\
            GROUP BY ba1.game_id, ba1.batter )"""
    )
    # found out DATE_SUB(loacl_date, 1) acts like a placeholder or count for the dates
    tableprep.show()

    print("CALCULATING 100 Day Rolling Average\n")
    # set up your input columns for calculation
    roll_avg_100 = R_AVG_100(
        inputCols=["TotalHits", "TotalAtBat"], outputCol="Rolling_Bat_AVG_100"
    )
    # this step calls your function from
    # Transformer_100_Day_AVG.py and does your calculation
    rolling = roll_avg_100.transform(tableprep)
    rolling.show()
    print("Showing 100 Day Rolling Average for each batter\n")


if __name__ == "__main__":
    sys.exit(main())
