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

package com.salesforce.hw.titanic

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

trait TitanicFeatures extends Serializable {

  val survived = FeatureBuilder.RealNN[Passenger].extract(_.getSurvived.toDouble.toRealNN).asResponse

  val pClass = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getPclass).map(_.toString).toPickList).asPredictor // scalastyle:off

  val name = FeatureBuilder.Text[Passenger].extract(d => Option(d.getName).toText).asPredictor

  val sex = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getSex).toPickList).asPredictor

  val age = FeatureBuilder.Real[Passenger].extract(d => Option(Double.unbox(d.getAge)).toReal).asPredictor

  val sibSp = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getSibSp).map(_.toString).toPickList).asPredictor

  val parch = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getParch).map(_.toString).toPickList).asPredictor

  val ticket = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getTicket).toPickList).asPredictor

  val fare = FeatureBuilder.Real[Passenger].extract(d => Option(Double.unbox(d.getFare)).toReal).asPredictor

  val cabin = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getCabin).toPickList).asPredictor

  val embarked = FeatureBuilder.PickList[Passenger].extract(d => Option(d.getEmbarked).toPickList).asPredictor

}
