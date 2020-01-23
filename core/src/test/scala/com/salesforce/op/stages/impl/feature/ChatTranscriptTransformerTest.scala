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

import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ChatTranscriptTransformerTest extends OpTransformerSpec[MapList, ChatTranscriptTransformer[Text]] {

  val chat1 = """<p align="center">Chat Started: Monday, December 17, 2018, 23:27:57 (-0500)</p><p align="center">
                |Chat Origin: Live Chat Button</p><p align="center">Agent Royalle G</p>( 10s )
                |Royalle G: Hi, this is Royalle with Hulu support. I&#39;ll be happy to assist you today,
                |before getting started may I have your email address and billing zip code please?<br>( 44s )
                |George Macaroni: Crystalizeforlife@yahoo.com 21209<br>( 1m 13s )
                |Royalle G: Thank you for verifying that information, one moment while I access your account.<br>
                |( 3m 20s ) Royalle G: I understand you are trying to watch on your Mac Mini, and there is a black
                |screen.<br>( 3m 29s ) George Macaroni: yes<br>( 3m 32s ) """".stripMargin

  val chat2 = Text.empty

  val chat3 = """<p align="center">Chat Started: Friday, September 20, 2019, 16:14:59 (+0000)</p><p align="center">
                |Chat Origin: Astro Chat</p><p align="center">Agent (UserFirstName)</p>( 5s )
                |Stephen Fountain: Hello, how can I help you today?<br>( 2m 24s ) Visitor: still connected<br>
                |""".stripMargin

  val chat4 = """<p align="center">Chat Started: Thursday, December 28, 2017, 21:36:39 (+0800)</p><p align="center">
      |Chat Origin: CRM Config Rollover</p><p align="center">Agent Zyr D</p>( 10s )
      |Zyr D: Hello Priscila Marques Fernandes, welcome to Salesforce.com chat. Thanks for your case information.
      |I am reviewing your inquiry now.<br>""".stripMargin

  val sample = Seq( Text(chat1) )

  val (inputData, f) = TestFeatureBuilder(sample)
  val transformer = new ChatTranscriptTransformer[Text]().setInput(f)

  val map1 = Map("speaker"->"Agent", "text" ->
    """ Hi, this is Royalle with Hulu support. I'll be happy to assist you today,before getting started
      |may I have your email address and billing zip code please?""".stripMargin)

  val map2 = Map("speaker"->"Customer", "text"->" Crystalizeforlife@yahoo.com 21209")

  val map3 = Map("speaker"->"Agent", "text"->
    """ Thank you for verifying that information, one moment while I access your account.  I understand you are trying
      |to watch on your Mac Mini, and there is a blackscreen.""".stripMargin)

  val map4 = Map("speaker"->"Customer", "text"->" yes")

  val expectedResult = Seq(new MapList(List(
    new TextMap(map1),
    new TextMap(map2),
    new TextMap(map3),
    new TextMap(map4))))
}
