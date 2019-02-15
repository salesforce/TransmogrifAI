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

package com.salesforce.op.utils.geo

/**
 * Reverse Geocoder trait
 */
trait ReverseGeocoder extends Serializable {

  /**
   * Find the nearest cities to the specified coordinate within the radius in KM
   *
   * @param latitude     latitude value
   * @param longitude    longitude value
   * @param radiusInKM   radius in KM
   * @param numOfResults number of results to return
   *
   * @return nearest cities to the specified coordinate within the radius in KM
   */
  def nearestCities(latitude: Double, longitude: Double, radiusInKM: Double, numOfResults: Int): Seq[WorldCity]

  /**
   * Find the nearest countries to the specified coordinate within the radius in KM
   *
   * @param latitude     latitude value
   * @param longitude    longitude value
   * @param radiusInKM   radius in KM
   * @param numOfResults number of results to return
   *
   * @return nearest countries to the specified coordinate within the radius in KM
   */
  def nearestCountries(latitude: Double, longitude: Double, radiusInKM: Double, numOfResults: Int): Seq[String]

}

/**
 * World City
 *
 * @param city       city name
 * @param country    country name
 * @param accentCity accent city name
 * @param region     region name
 * @param population population
 * @param latitude   latitude value
 * @param longitude  longitude value
 */
case class WorldCity
(
  city: String,
  country: String,
  accentCity: String,
  region: String,
  population: Long,
  latitude: Double,
  longitude: Double
)
