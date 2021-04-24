var widgets = require('@jupyter-widgets/base');
var _ = require('lodash');
var $ = require('jquery');
require("./style.css");
require("./magnific-popup.css");
require('magnific-popup');

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.


function generate_div_for_data(data, selection_callback) {
	/*
	Construct the display html for the given list of images + code to display.
	The `data` argument should be a list of records (dictionaries) with every record having at least the following entries:
	{
		'png': The png corresponding to the visualization in base64 format,
		'code_html': HTML representation of the code. You can use the pygments library on the Python side to format code,
		'idx': An integer uniquely identifying the element. This must be unique across all elements across all pages.
	}
	 */
	let img_elems = [];
	let div_elems = [];
	let popup_elems = [];
	let id = 1;
	data.forEach(elem => {
		let img_src = 'data:image/png;base64,' + elem.png
		// The following is the thumbnail seen on the jupyter notebook output
		let img_elem = $('' +
			'<div>' +
			'<img src="' + img_src + '">' +
			'<div class="img-hover-button-div">' +
			'<a href="#viz-widget-popup-' + id + '" class="btn btn-success mybtn-confirm">Select' +
			'<a href="#viz-widget-popup-' + id + '" class="btn btn-default mybtn-expand">Expand' +
			'</div>' +
			'</a>' + '</div>');

		let code_html = "";
		for (let i = 0; i < elem.code_html.length; i++) {
			code_html += '<div class="code-container">' +
            elem.code_html[i] +
			'</div>';
		}

		// This controls the display when the thumbnail is clicked to open the popup
		let popup_elem_img = $('' +
			'<div class="mfp-hide" id="viz-widget-popup-' + id + '">' +
			// Select Button
			'<div style="text-align: center; padding: 20px">' +
			'<button style="margin-right:16px" class="btn btn-success btn-lg mybtn-confirm">Select</button>' +
			'<button class="btn btn-default btn-lg mybtn-show-code">Show Code(s)</a>' +
			'<button class="mfp-hide btn btn-default btn-lg mybtn-hide-code">Hide Code(s)</a>' +
			'</div>' +

			// Image
			'<div class="viz-widget-popup-img">' +
			'<img src="' + img_src + '">' +
            '</div>' +

			// Code
			'<div class="mfp-hide viz-widget-popup-code" style="text-align: center">' +
            code_html +
			'</div>' +
			'</div>');

		img_elems.push(img_elem[0]);
		popup_elems.push(popup_elem_img[0]);

		// Assemble the elements.
		let div_elem = document.createElement("div");
		let img_container_elem = document.createElement("div");
		img_container_elem.appendChild(img_elem[0]);
		img_container_elem.setAttribute("class", "viz-img-container")
		div_elem.setAttribute("class", "viz-container");
		div_elem.appendChild(img_container_elem);
		div_elem.appendChild(popup_elem_img[0]);
		div_elems.push(div_elem);
		id = id + 1;

		// If the select button is clicked, the Python side callback should be triggered.
		$(".mybtn-confirm", popup_elem_img).click(function () {
		    selection_callback(elem['idx']);
			$.magnificPopup.instance.close();
		});

		$(".mybtn-confirm", img_elem).click(function () {
			selection_callback(elem['idx']);
		});
	});

	// The library code that connects the popup to the thumbnails
	$(".mybtn-expand", $(img_elems)).magnificPopup({
		type: 'inline',
		gallery: {
			enabled: true,
		},
        closeBtnInside: false,
	});

	let j_popup_elems = $(popup_elems);
	console.log(j_popup_elems);

	// Toggle between showing and hiding code across all popups.
	$(".mybtn-show-code", j_popup_elems).click(function () {
		$(".viz-widget-popup-code", j_popup_elems).removeClass("mfp-hide");
		$(".mybtn-show-code", j_popup_elems).addClass("mfp-hide");
		$(".mybtn-hide-code", j_popup_elems).removeClass("mfp-hide");
	});

	$(".mybtn-hide-code", j_popup_elems).click(function () {
		$(".viz-widget-popup-code", j_popup_elems).addClass("mfp-hide");
		$(".mybtn-show-code", j_popup_elems).removeClass("mfp-hide");
		$(".mybtn-hide-code", j_popup_elems).addClass("mfp-hide");
	});


	return div_elems;
}

// Custom View. Renders the widget model.

var VizSynthesisWidgetModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'VizSynthesisWidgetModel',
        _view_name : 'VizSynthesisWidgetView',
        _model_module : 'viz_synthesis_widget',
        _view_module : 'viz_synthesis_widget',
        _model_module_version : '^0.1.0',
        _view_module_version : '^0.1.0',
        data: [],
        num_cols: 3,
		selection: -1,
    })
});

var VizSynthesisWidgetView = widgets.DOMWidgetView.extend({
	render: function () {
		this.data_changed();
		this.model.on('change:data', this.data_changed, this);
		this.model.on('change:num_cols', this.num_cols_changed, this);
	},

	selection_callback: function (idx) {
		this.model.set({
			'selection': idx,
		});
		this.touch();
		console.log('Selected ' + idx);
	},

	data_changed: function () {
		while (this.el.firstChild) {
			this.el.removeChild(this.el.firstChild);
		}

		let view = this;
	    let data = this.model.get('data');
		let divs = generate_div_for_data(data, (idx => view.selection_callback(idx)));
		this.section = document.createElement('section');
		this.section.setAttribute("class", "photos");
		divs.forEach(e => {
			this.section.appendChild(e);
		});

		this.section.style.columnCount = this.model.get('num_cols');
		this.el.appendChild(this.section);
	},

	num_cols_changed: function () {
		this.section.style.columnCount = this.model.get('num_cols');
	},
});

module.exports = {
    VizSynthesisWidgetModel : VizSynthesisWidgetModel,
    VizSynthesisWidgetView : VizSynthesisWidgetView
};
