/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import scala.io.Source
import scala.util.Random

/**
 * References a bunch of geographic, statistical etc data.
 *
 * See testkit/README.md in resources for the origins of data
 */
object DataSources {

  /**
   * Capitalize each word here
   *
   * @param s source, e.g. UNITED EMIRATES
   * @return each word capitalized, e.g. United Emirates
   */
  private def extract(s: String): String =
    s.split("[,\\(]").head.trim.toLowerCase.split(" ").map(_.capitalize).mkString(" ")

  private def namesFrom(strings: Iterator[String]): List[String] = strings.map(extract).toList

  def linesFromResource(path: String): Iterator[String] = {
    val isOpt = Option(getClass.getResourceAsStream(path))
    isOpt map (is => Source.fromInputStream(is).getLines) getOrElse {
      throw new ExceptionInInitializerError(s"Resource <<$path>> not found.")
    }
  }

  private def namesFromResourceFile(path: String) = namesFrom(linesFromResource(path))

  val LastNames: List[String] = namesFromResourceFile("/lastnames.txt")

  val FirstNames: List[String] = namesFromResourceFile("/firstnames.txt")

  val RealCountries: List[String] = namesFromResourceFile("/countries.txt")

  val ImaginaryCountries: List[String] = namesFromResourceFile("/imaginaryCountries.txt")

  val Countries: List[String] = (RealCountries ++ ImaginaryCountries) filter (_.nonEmpty)

  val States: List[String] = namesFromResourceFile("/states.txt")

  val CitiesOfCalifornia: List[String] = namesFromResourceFile("/cities.txt")

  val StreetsOfSanJose: List[String] = namesFromResourceFile("/streets.txt")

  val RandomFirstName: Random => String = RandomStream of FirstNames

  val RandomLastName: Random => String = RandomStream of LastNames

  val RandomName: Random => String = rng => RandomFirstName(rng) + " " + RandomLastName(rng)

  // source: [[https://www.nationalnanpa.com/nanp1/npa_report.csv]]
  // @see [[https://www.nationalnanpa.com/area_codes/]]
  private def areacodesFromResourceFile: List[String] = {
    val GoodCountry = Set("CA", "US")
    def isAcceptable(line: String) = !line.contains("Suspended") && !line.contains("Effective with completion")

    val lines = linesFromResource("/npa_report.csv")
    //                                                 Assigned,...............Country,In Service
    val IsAssigned = "(\\d\\d\\d),[^,]*,[^,]*,[^,]*,[^,]*,Yes,[^,]*,[^,]*,[^,]*,([^,]*),Y.*".r
    val goodNumbers = lines.toList collect {
      case line@IsAssigned(n, c) if GoodCountry(c) && isAcceptable(line) => n
    }
    goodNumbers
  }

  val ValidAreaCodes: List[String] = areacodesFromResourceFile

  val RandomAreaCode: Random => String = RandomStream of ValidAreaCodes

}
