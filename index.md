---
# You don't need to edit this file, it's empty on purpose.
# Edit theme's home layout instead if you wanna make some changes
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: page
title: Home
---
<br />
<h3>Introduction</h3>
<p>
This is a collection of Python machine learning examples that I'm trying to get
my head around. Attribution is provided with many thanks.
</p>
<h3>Recent Posts</h3>
<ul>
    {% for post in site.posts limit:5 %}
      <li><a href="{{ site.baseurl}}{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
</ul>
