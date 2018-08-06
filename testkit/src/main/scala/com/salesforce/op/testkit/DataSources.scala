/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
