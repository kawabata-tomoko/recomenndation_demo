import secrets
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from src.model import RecommendationCollection
global RECOMMENTATION_NUMBER
RECOMMENTATION_NUMBER=10
app = Flask(__name__)

app.secret_key = secrets.token_hex(
    16
)  # 设置一个密钥用于加密 session 数据，可以是随机的字符串

# 初始化推荐系统
RCM = RecommendationCollection(
    "./src/dataset/dataset.csv", datasetArgs=dict(x=2 / 3, random=False)
)


# 登录页面路由
@app.route("/login", methods=["GET", "POST"])
def login():
    # 清空 session
    session.clear()

    if request.method == "POST":
        # 获取用户输入的ID
        user_id = request.form.get("userID")

        # 用户ID的验证
        if not str(user_id) in RCM.meta_user.index.astype(str).values:
            error_message = "无效的用户id, 请输入正确的用户id"
            return render_template("login.html", error_message=error_message)

        # 用户ID有效
        # 获取用户信息
        user_info = RCM.get_user_info(user_id)
        # 将用户信息存储在session中
        session["user_info"] = user_info
        # 重定向到推荐页面，并传递用户ID
        return redirect(url_for("index", user_id=user_id))

    # 如果是GET请求，渲染登录页面
    return render_template("login.html")


# 主页路由
@app.route("/")
def index():
    # 获取从登录页面传递过来的用户ID，如果没有则重定向到登录页面
    user_id = request.args.get("user_id")
    if user_id is None:
        return redirect(url_for("login"))

    # 在这里处理获取用户信息的逻辑
    user_info = RCM.get_user_info(user_id)

    # 获取当前页数，默认为第一页
    page = request.args.get("page", 1, type=int)
    page_size = 9

    # 分页查询课程
    courses_for_page = RCM.get_courses_for_page(page, page_size)

    # 计算总页数
    total_pages = courses_for_page["total_pages"]
    courses = courses_for_page["courses"]
    # 获取当前页的前后各5个页码
    start_page = max(1, page - 5)
    end_page = min(total_pages, page + 5)

    # 渲染模板，并传递分页信息和课程数据
    return render_template(
        "index.html",
        user_info=user_info,
        courses=courses,
        page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page,
    )


# 用户已学课程页面路由
@app.route("/user_courses_history/<user_id>")
def user_courses_history(user_id):
    # 获取用户信息
    user_info = session.get("user_info")

    user_courses_history = RCM.get_user_courses_history(user_id)

    # 渲染已学课程页面，
    return render_template(
        "user_courses_history.html",
        user_info=user_info,
        user_courses_history=user_courses_history,
    )


# 基于协同过滤的学习资源推荐
@app.route("/personalized_recommendation/<user_id>")
def personalized_recommendation(user_id):
    # 在这里处理获取用户信息的逻辑
    user_info = RCM.get_user_info(user_id)
    RCM.init_model(
        "CFM",
        modelArgs=dict(
            n_reference=20,  # 协同节点数
            transpose=True,  # 如果需要基于用户协同过滤请改为False
        ),
    )  # 默认使用基于课程的协同过滤
    # 为用户生成推荐
    recommendations = RCM.get_course_info_by_id_list(
        RCM.recommend(
            user_id,
            n_recommendations=RECOMMENTATION_NUMBER,  # 推荐课程数
            recommend_args=dict(
                axis=1,  # 聚合维度,基于用户协同过滤请改为0
                force=False,  # 数据缺失时强制输出结果
                keep_learnt=True,  # 允许推荐已学习课程
                )
        )
    )

    # 渲染个性化推荐页面，并传递推荐信息和用户ID
    return render_template(
        "personalized_recommendation.html",
        user_info=user_info,
        recommendations=recommendations,
    )


# 基于聚类的学习资源推荐
@app.route("/similar_users_recommendation/<user_id>")
def similar_users_recommendation(user_id):
    # 获取用户信息
    user_info = session.get("user_info")
    RCM.init_model(
        "CRM",
        modelArgs=dict(
            transpose=False,  # 请不要修改此项,否则会变为为课程推荐用户
            cluser_args={
                "t1": 6.5,
                "t2": 6,
            },  # 实验确定Canopy参数,预计生成约100个左右的中心点
        ),
    )
    # 假设有一个名为 similar_books_data 的数据，包含相似用户推荐的课程信息
    recommendations = RCM.get_course_info_by_id_list(
        RCM.recommend(
            user_id,
            n_recommendations=RECOMMENTATION_NUMBER  # 推荐课程数
        )
    )

    return render_template(
        "similar_users_recommendation.html",
        user_info=user_info,
        recommendations=recommendations,
    )


@app.route("/search_courses")
def search_courses():
    # 获取用户信息
    user_info = session.get("user_info")

    # 获取搜索关键词
    query = request.args.get("query", "")

    # 执行搜索逻辑，可以在这里调用你的搜索函数
    search_results = RCM.search_courses_by_name(query)

    # 渲染搜索结果页面
    return render_template(
        "search_results.html",
        query=query,
        search_results=search_results,
        user_info=user_info,
    )


# 所有用户信息页面路由
@app.route("/all_users")
def all_users():
    # 获取所有用户信息
    # 获取当前页数，默认为第一页
    page = request.args.get("page", 1, type=int)
    page_size = 9

    # 分页查询课程
    users_for_page = RCM.get_user_for_page(page, page_size)

    # 计算总页数
    total_pages = users_for_page["total_pages"]
    users = users_for_page["users"]
    # 获取当前页的前后各5个页码
    start_page = max(1, page - 5)
    end_page = min(total_pages, page + 5)

    # 渲染模板，并传递分页信息和课程数据
    return render_template(
        "all_users.html",
        users=users,
        page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page,
    )
    # # 渲染显示所有用户信息的页面
    # return render_template("all_users.html", all_user_info=all_user_info)


if __name__ == "__main__":
    app.run(debug=True)
