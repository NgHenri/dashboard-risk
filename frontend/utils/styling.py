# frontend/utils/styling.py

from st_aggrid.shared import JsCode

def build_dynamic_styling(style_rules):
    full_js = "function(params) {\n"
    for libelle, js_func in style_rules.items():
        full_js += f"""
        if (params.data['Libellé'] === '{libelle}') {{
            return ({js_func.js_code})(params);
        }}
        """
    full_js += "return {};\n}"
    return JsCode(full_js)

style_rules = {
    "Part des mensualités dans les revenus": JsCode("""
        function(params) {
            const value = parseFloat(params.value.replace('%', '').replace(',', '.'));
            if (isNaN(value)) return {};
            if (value < 10) {
                return { 'color': 'white', 'backgroundColor': '#4CAF50' };
            } else if (value < 30) {
                return { 'color': 'black', 'backgroundColor': '#FFC107' };
            } else {
                return { 'color': 'white', 'backgroundColor': '#F44336' };
            }
        }
    """),
    "Ratio crédit/revenu": JsCode("""
        function(params) {
            const value = parseFloat(params.value.replace('%', '').replace(',', '.'));
            if (isNaN(value)) return {};
            if (value < 30) {
                return { 'color': 'white', 'backgroundColor': '#4CAF50' };
            } else if (value < 60) {
                return { 'color': 'black', 'backgroundColor': '#FFC107' };
            } else {
                return { 'color': 'white', 'backgroundColor': '#F44336' };
            }
        }
    """),
    # Ajoute ici d'autres règles si besoin...
}
