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
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer

import java.util.regex.Pattern
import java.lang.IndexOutOfBoundsException

import scala.collection.mutable.MutableList
import scala.reflect.runtime.universe.TypeTag
import scala.collection.JavaConverters._

import org.jsoup.Jsoup

/** the Html parser is selected based on https://stackoverflow.com/questions/2168610/which-html-parser-is-the-best
 */
class ChatTranscriptTransformer[I <: Text](uid: String = UID[ChatTranscriptTransformer[_]])(implicit tti: TypeTag[I])
  extends UnaryTransformer[I, MapList](operationName = "chatParsing", uid = uid) {

  override def transformFn: I => MapList = (chats: I) => {
    if ( !chats.isEmpty ) {
      val elements = Jsoup.parse( chats.value.get ).body().getElementsByTag("p").eachText().asScala.toList
      val agent = ChatTranscriptParser.getAgent( elements )

      val texts = Jsoup.parse( chats.value.get ).body().textNodes().asScala.toList
      val parsedChats = texts.map( line => ChatTranscriptParser.cleanUtterance(line.text(), agent) )
                             .filter( !_.isEmpty )
      ChatTranscriptParser.combineSpeakers( parsedChats )
    }
    else MapList.empty
  }
}

  /** in cleanUtterrance,  matcher.matches() or matcher.find() needs to be called to allow group()
   *  Stack overflow post: http://tiny.cc/dpm0iz
   */
object ChatTranscriptParser {

  val agentExpr = Pattern.compile("""Agent ([\w\s]+)""")
  val chatBotExpr = Pattern.compile("""Chatbot successfully""")
  val responseTimeExpr = Pattern.compile("""\( [\w\s]+ \)""")

  /**
   * find lines matching 'Agent ([\w\s])+)' but not 'Chatbot succssfully'. return agentname
   */
  def getAgent(chats: List[String]): Option[String] = {
    val chatContainingAgentName = chats.filter(utterance => {
      val mc = chatBotExpr.matcher(utterance)
      val ma = agentExpr.matcher(utterance)
      (!mc.lookingAt()) && (ma.lookingAt()) })

    if(chatContainingAgentName.size>0) {
      val m = agentExpr.matcher(chatContainingAgentName(0))
      m.find()
      Some(m.group(1))
    } else {
      None
    }
  }

  /**
   * split utterances into speaker/text key-value TextMap
   */
  def cleanUtterance(utterance: String, agentName: Option[String]): TextMap = {
    try {
      val cleanedUtterance = responseTimeExpr.matcher(utterance).replaceAll("")
      val speaker = cleanedUtterance.split(":")(0)
      val chatText = cleanedUtterance.split(":")(1)

      if ( speaker.contains(agentName.getOrElse("n/a")) ) TextMap.apply(Map("speaker" -> "Agent", "text" -> chatText))
      else TextMap.apply(Map("speaker" -> "Customer", "text" -> chatText))
    } catch {
      case e: IndexOutOfBoundsException => TextMap.empty
    }
  }

  /** Combines all consecutive messages by 1 speaker into a single message.
   *  For example, if
   * 1. Customer sends 2 consecutive messages.
   * 2. Agent sends 3 consecutive messages.
   * then 2 messages (1 customer, and 1 agent) will be returned.
   * Messages are concatenated with a space delimiter.
   */
  def combineSpeakers(speakerTextPairs: List[TextMap]): MapList = {
    if (speakerTextPairs.size<=1 ) MapList.apply(speakerTextPairs)
    else {
      var outputList = MutableList(speakerTextPairs(0))
      for{ element <- speakerTextPairs.slice(1, speakerTextPairs.size)} {
        val lastElement = outputList.last.value
        if (element.value.getOrElse("speaker", "").contentEquals( lastElement.getOrElse("speaker", "")) ) {
          val updatedMap = lastElement.updated("text",
            lastElement.getOrElse("text", "") + " " + element.value.getOrElse("text", ""))
          outputList = outputList.dropRight(1) ++ List( TextMap.apply(updatedMap) )
        }
        else outputList = outputList ++ List( element )
      }
      MapList.apply(outputList)
    }
  }
}



