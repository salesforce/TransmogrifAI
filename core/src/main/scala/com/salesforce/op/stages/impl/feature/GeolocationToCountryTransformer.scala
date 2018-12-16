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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{Geolocation, Text}
import com.salesforce.op.stages.base.unary.UnaryTransformer
import geocode.ReverseGeoCode

class GeolocationToCountryTransformer(uid: String = UID[GeolocationToCountryTransformer])
  extends UnaryTransformer[Geolocation, Text](operationName = "geoToCountry", uid = uid) {

  override def transformFn: Geolocation => Text = (geolocation: Geolocation) => {
    if (geolocation == null || geolocation.lat == null || geolocation.lon == null) {
      new Text("")
    }

    val lat: Double = geolocation.lat
    val long: Double = geolocation.lon

    new Text(getCountryName(lat, long))
  }

  private[op] def getCountryName(lat: Double, long: Double): String = {
    GeolocationToCountryTransformer.reverseGeoCode.nearestPlace(lat, long).country
  }
}

object GeolocationToCountryTransformer {
  val reverseGeoCode = new ReverseGeoCode(
    getClass.getClassLoader.getResourceAsStream("geolocation/cities1000.txt"), true)
}
