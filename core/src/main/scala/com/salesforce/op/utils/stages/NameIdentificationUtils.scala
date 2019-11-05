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

package com.salesforce.op.utils.stages

import com.salesforce.op.utils.json.JsonLike

import scala.io.Source
import scala.util.Try

/**
 * * Documentation for the sources of name data, configuration settings
 */
object NameIdentificationUtils {
  case class NameDictionary
  (
    value: Set[String] = {
      val nameDictionary = collection.mutable.Set.empty[String]
      val dictionaryPath = "/NameIdentification_JRC.txt"
      val stream = getClass.getResourceAsStream(dictionaryPath)
      val buffer = Source.fromInputStream(stream)
      for {name <- buffer.getLines} {
        nameDictionary += name
      }
      buffer.close
      nameDictionary.toSet[String]
    }
  ) extends JsonLike

  case class GenderDictionary
  (
    value: Map[String, Double] = {
      val genderDictionary = collection.mutable.Map.empty[String, Double]
      val dictionaryPath = "/GenderDictionary_SSA.csv"
      val stream = getClass.getResourceAsStream(dictionaryPath)
      val buffer = Source.fromInputStream(stream)
      // TODO: Also make use of frequency information in this dictionary
      for {row <- buffer.getLines.drop(1)} {
        val cols = row.split(",").map(_.trim)
        val name = cols(0).toLowerCase().replace("\\P{L}", "")
        val probMale = Try {
          cols(6).toDouble
        }.toOption
        probMale match {
          case Some(prob) => genderDictionary += (name -> prob)
          case None =>
        }
      }
      buffer.close
      genderDictionary.toMap[String, Double]
    }
  ) extends JsonLike

  val TokensToCheckForFirstName: Seq[Int] = Seq(0, -1)
  val EmptyTokensMap: Map[Int, Int] = Map(TokensToCheckForFirstName map { i => i -> 0 }: _*)
}
