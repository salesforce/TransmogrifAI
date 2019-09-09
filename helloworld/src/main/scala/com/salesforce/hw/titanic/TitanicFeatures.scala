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

import java.io.Serializable

import com.salesforce.hw.titanic.TitanicFeatures._
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

trait TitanicFeatures extends Serializable {
  val survived = FeatureBuilder.RealNN[Passenger].extract(new SurvivedFunc).asResponse
  val pClass = FeatureBuilder.PickList[Passenger].extract(new pClassFunc).asPredictor // scalastyle:off
  val name = FeatureBuilder.Text[Passenger].extract(new NameFunc).asPredictor
  val sex = FeatureBuilder.PickList[Passenger].extract(new SexFunc).asPredictor
  val age = FeatureBuilder.Real[Passenger].extract(new AgeFunc).asPredictor
  val sibSp = FeatureBuilder.PickList[Passenger].extract(new SibSpFunc).asPredictor
  val parch = FeatureBuilder.PickList[Passenger].extract(new ParchFunc).asPredictor
  val ticket = FeatureBuilder.PickList[Passenger].extract(new TicketFunc).asPredictor
  val fare = FeatureBuilder.Real[Passenger].extract(new FareFunc).asPredictor
  val cabin = FeatureBuilder.PickList[Passenger].extract(new CabinFunc).asPredictor
  val embarked = FeatureBuilder.PickList[Passenger].extract(new EmbarkedFunc).asPredictor
}

object TitanicFeatures {
  class SurvivedFunc extends Function1[Passenger, RealNN] with Serializable {
    override def apply(v1: Passenger): RealNN = v1.getSurvived.toDouble.toRealNN
  }

  class pClassFunc extends Function1[Passenger, PickList] with Serializable {
    override def apply(v1: Passenger): PickList = Option(v1.getPclass).map(_.toString).toPickList
  }

  class NameFunc extends Function1[Passenger, Text] with Serializable {
    override def apply(v1: Passenger): Text = Option(v1.getName).toText
  }

  class SexFunc extends Function1[Passenger, PickList] with Serializable {
    override def apply(v1: Passenger): PickList = Option(v1.getSex).toPickList
  }

  class AgeFunc extends Function1[Passenger, Real] with Serializable {
    override def apply(v1: Passenger): Real = Option(Double.unbox(v1.getAge)).toReal
  }

  class SibSpFunc extends Function1[Passenger, PickList] with Serializable {
    override def apply(v1: Passenger): PickList = Option(v1.getSibSp).map(_.toString).toPickList
  }

  class ParchFunc extends Function1[Passenger, PickList] with Serializable {
    override def apply(v1: Passenger): PickList = Option(v1.getParch).map(_.toString).toPickList
  }

  class TicketFunc extends Function1[Passenger, PickList] with Serializable {
    override def apply(v1: Passenger): PickList = Option(v1.getTicket).toPickList
  }

  class FareFunc extends Function1[Passenger, Real] with Serializable {
    override def apply(v1: Passenger): Real = Option(Double.unbox(v1.getFare)).toReal
  }

  class CabinFunc extends Function1[Passenger, PickList] with Serializable {
    override def apply(v1: Passenger): PickList = Option(v1.getCabin).toPickList
  }

  class EmbarkedFunc extends Function1[Passenger, PickList] with Serializable {
    override def apply(v1: Passenger): PickList = Option(v1.getEmbarked).toPickList
  }
}
