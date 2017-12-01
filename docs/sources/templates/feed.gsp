<% import static groovy.xml.XmlUtil.escapeXml %><?xml version="1.0"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>JBake</title>
    <link>${config.site_host}</link>
    <atom:link href="${config.site_host}/${config.feed_file}" rel="self" type="application/rss+xml" />
    <description>JBake Bootstrap Template</description>
    <language>en-gb</language>
    <pubDate>${new java.text.SimpleDateFormat("EEE, d MMM yyyy HH:mm:ss Z", Locale.ENGLISH).format(published_date)}</pubDate>
    <lastBuildDate>${new java.text.SimpleDateFormat("EEE, d MMM yyyy HH:mm:ss Z", Locale.ENGLISH).format(published_date)}</lastBuildDate>

    <%published_posts.each {post -> %>
    <item>
      <title>${post.title}</title>
      <link>${config.site_host}/${post.uri}</link>
      <pubDate>${new java.text.SimpleDateFormat("EEE, d MMM yyyy HH:mm:ss Z", Locale.ENGLISH).format(post.date)}</pubDate>
      <guid isPermaLink="false">${post.uri}</guid>
      <description>
      ${escapeXml(post.body)}
	  </description>
    </item>
    <%}%>

  </channel> 
</rss>
